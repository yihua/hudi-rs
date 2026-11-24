/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Reads the same file slices through hudi-rs's `FileGroupReader` and through
//! the parquet crate's async reader, with the same projection and batch size,
//! and reports CPU, wall time, rows, allocations and peak RSS per mode. An
//! optional pprof profile of the last iteration prints its hottest frames.
//!
//! Modes:
//! - `hudi`: the slice as the native Spark scan reads it (base file plus logs).
//! - `hudi-base`: the same reader with the logs left out (base-file stream path).
//! - `parquet`: the base files read directly with `ParquetRecordBatchStreamBuilder`
//!   over the same object store.

mod rusage;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use clap::Parser;
use futures::{StreamExt, TryStreamExt};
use hudi_core::config::read_options::ReadOptions;
use hudi_core::table::builder::TableBuilder;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use parquet::arrow::ProjectionMask;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};

struct Counting;

static COUNT_ON: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT_ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNT_ON.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[derive(Parser, Debug)]
struct Args {
    /// Absolute local path of the Hudi table.
    #[arg(long)]
    table: String,
    /// Partition directories (relative to the table) whose slices are read.
    #[arg(long, value_delimiter = ',')]
    partitions: Vec<String>,
    /// Columns to project.
    #[arg(long, value_delimiter = ',')]
    columns: Vec<String>,
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,
    /// hudi | hudi-base | parquet
    #[arg(long, default_value = "hudi")]
    mode: String,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Slices read at once.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    /// Count allocations (single-threaded runs only, the counters are shared).
    #[arg(long)]
    count_allocs: bool,
    /// Write a flamegraph SVG of the last iteration here and print its hottest frames.
    #[arg(long)]
    profile: Option<String>,
    /// Extra `hoodie.*` read options, `k=v`.
    #[arg(long, value_delimiter = ',')]
    hudi_opt: Vec<String>,
}

#[derive(Clone, Debug)]
struct Slice {
    base: String,
    base_size: u64,
    logs: Vec<String>,
}

/// Latest base file per file id plus the log files written after it. Good enough
/// for a benchmark over a table without pending compactions or replaced groups.
fn slices(table: &str, partitions: &[String]) -> Vec<Slice> {
    let mut out = Vec::new();
    for part in partitions {
        let dir = std::path::Path::new(table).join(part);
        let mut bases: HashMap<String, (String, String, u64)> = HashMap::new();
        let mut logs: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for entry in std::fs::read_dir(&dir).expect("partition dir") {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".parquet") {
                let file_id = stem.split('_').next().unwrap().to_string();
                let instant = stem.rsplit('_').next().unwrap().to_string();
                let size = entry.metadata().unwrap().len();
                let newer = bases.get(&file_id).is_none_or(|(i, _, _)| *i < instant);
                if newer {
                    bases.insert(file_id, (instant, name, size));
                }
            } else if name.starts_with('.') && name.contains(".log.") {
                let rest = &name[1..];
                let (head, _) = rest.split_once(".log.").unwrap();
                let (file_id, instant) = head.rsplit_once('_').unwrap();
                logs.entry(file_id.to_string())
                    .or_default()
                    .push((instant.to_string(), name.clone()));
            }
        }
        for (file_id, (base_instant, name, size)) in bases {
            let mut l: Vec<(String, String)> = logs
                .remove(&file_id)
                .unwrap_or_default()
                .into_iter()
                .filter(|(i, _)| *i > base_instant)
                .collect();
            l.sort();
            out.push(Slice {
                base: format!("{part}/{name}"),
                base_size: size,
                logs: l.into_iter().map(|(_, n)| format!("{part}/{n}")).collect(),
            });
        }
    }
    out.sort_by(|a, b| a.base.cmp(&b.base));
    out
}

async fn run_hudi(args: &Args, slices: &[Slice], with_logs: bool) -> (u64, usize) {
    let mut opts: Vec<(String, String)> = Vec::new();
    for kv in &args.hudi_opt {
        let (k, v) = kv.split_once('=').expect("k=v");
        opts.push((k.to_string(), v.to_string()));
    }
    let table = TableBuilder::from_base_uri(&format!("file://{}", args.table))
        .with_options(opts.clone())
        .build()
        .await
        .expect("open table");
    let read_options = ReadOptions::new()
        .with_hudi_options(opts.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .with_projection(args.columns.clone())
        .with_batch_size(args.batch_size)
        .unwrap();
    let reader = Arc::new(
        table
            .create_file_group_reader_with_options(
                Some(&read_options),
                std::iter::empty::<(&str, String)>(),
            )
            .await
            .expect("reader"),
    );
    let results: Vec<(u64, usize)> = futures::stream::iter(slices.iter().cloned())
        .map(|s| {
            let reader = reader.clone();
            let read_options = read_options.clone();
            async move {
                let logs: Vec<String> = if with_logs { s.logs.clone() } else { vec![] };
                let mut stream = reader
                    .read_file_slice_from_paths_stream_with_base_file_size(
                        &s.base,
                        Some(s.base_size),
                        logs.iter(),
                        &read_options,
                    )
                    .await
                    .expect("open slice");
                let (mut rows, mut batches) = (0u64, 0usize);
                while let Some(b) = stream.next().await {
                    let b = b.expect("batch");
                    rows += b.num_rows() as u64;
                    batches += 1;
                }
                (rows, batches)
            }
        })
        .buffer_unordered(args.concurrency)
        .collect()
        .await;
    results
        .into_iter()
        .fold((0, 0), |(r, b), (r2, b2)| (r + r2, b + b2))
}

async fn run_parquet(args: &Args, slices: &[Slice]) -> (u64, usize) {
    let store: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&args.table).expect("local store"));
    let results: Vec<(u64, usize)> = futures::stream::iter(slices.iter().cloned())
        .map(|s| {
            let store = store.clone();
            async move {
                let reader = ParquetObjectReader::new(store, ObjectPath::from(s.base.as_str()))
                    .with_file_size(s.base_size);
                let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
                let indices: Vec<usize> = args
                    .columns
                    .iter()
                    .map(|c| builder.schema().index_of(c).expect("column"))
                    .collect();
                let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
                let mut stream = builder
                    .with_projection(mask)
                    .with_batch_size(args.batch_size)
                    .build()
                    .unwrap();
                let (mut rows, mut batches) = (0u64, 0usize);
                while let Some(b) = stream.try_next().await.unwrap() {
                    rows += b.num_rows() as u64;
                    batches += 1;
                }
                (rows, batches)
            }
        })
        .buffer_unordered(args.concurrency)
        .collect()
        .await;
    results
        .into_iter()
        .fold((0, 0), |(r, b), (r2, b2)| (r + r2, b + b2))
}

fn short(name: &str) -> String {
    let n = name.split("::h").next().unwrap_or(name);
    let n = if n.len() > 150 { &n[..150] } else { n };
    n.to_string()
}

fn print_profile(report: &pprof::Report, top: usize) {
    let mut total: i64 = 0;
    let mut self_counts: HashMap<String, i64> = HashMap::new();
    let mut incl: HashMap<String, i64> = HashMap::new();
    for (frames, count) in report.data.iter() {
        let count = *count as i64;
        total += count;
        let names: Vec<String> = frames
            .frames
            .iter()
            .flat_map(|f| f.iter().map(|s| short(&s.name())))
            .collect();
        if let Some(leaf) = names.first() {
            *self_counts.entry(leaf.clone()).or_default() += count;
        }
        let mut seen = std::collections::HashSet::new();
        for n in names {
            if seen.insert(n.clone()) {
                *incl.entry(n).or_default() += count;
            }
        }
    }
    let mut s: Vec<_> = self_counts.into_iter().collect();
    s.sort_by(|a, b| b.1.cmp(&a.1));
    println!("PROFILE samples={total}");
    println!("PROFILE top self:");
    for (n, c) in s.iter().take(top) {
        println!("PROFILE self {:5.1}% {}", 100.0 * *c as f64 / total as f64, n);
    }
    let mut i: Vec<_> = incl.into_iter().collect();
    i.sort_by(|a, b| b.1.cmp(&a.1));
    println!("PROFILE top inclusive:");
    for (n, c) in i
        .iter()
        .filter(|(n, _)| !BORING.iter().any(|b| n.contains(b)))
        .take(top * 2)
    {
        println!("PROFILE incl {:5.1}% {}", 100.0 * *c as f64 / total as f64, n);
    }
    // Stacks, leaf first, cut to the first few frames that name a function of
    // interest, so a hot leaf (an atomic add, a memcpy) shows who called it.
    let mut stacks: HashMap<String, i64> = HashMap::new();
    for (frames, count) in report.data.iter() {
        let names: Vec<String> = frames
            .frames
            .iter()
            .flat_map(|f| f.iter().map(|s| short_fn(&s.name())))
            .filter(|n| !BORING.iter().any(|b| n.contains(b)))
            .take(8)
            .collect();
        *stacks.entry(names.join(" <- ")).or_default() += *count as i64;
    }
    let mut st: Vec<_> = stacks.into_iter().collect();
    st.sort_by(|a, b| b.1.cmp(&a.1));
    println!("PROFILE top stacks:");
    for (n, c) in st.iter().take(top + 15) {
        println!("PROFILE stack {:5.1}% {}", 100.0 * *c as f64 / total as f64, n);
    }
}

const BORING: &[&str] = &[
    "poll",
    "catch_unwind",
    "call_once",
    "block_on",
    "lang_start",
    "_start",
    "main",
    "with_budget",
    "budget",
    "enter_runtime",
    "async_block",
    "async_fn",
    "{closure",
    "__rust_begin_short_backtrace",
    "try_next",
    "next_inner",
    "into_future",
];

/// A symbol without generics, cut to its last two path segments, so stacks stay short.
fn short_fn(name: &str) -> String {
    let n = name.split("::h").next().unwrap_or(name);
    let mut depth = 0i32;
    let mut plain = String::new();
    for ch in n.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ if depth == 0 => plain.push(ch),
            _ => {}
        }
    }
    let parts: Vec<&str> = plain.split("::").filter(|p| !p.is_empty()).collect();
    let k = parts.len().saturating_sub(2);
    parts[k..].join("::")
}

fn main() {
    let args = Args::parse();
    let slices = slices(&args.table, &args.partitions);
    let n_logs: usize = slices.iter().map(|s| s.logs.len()).sum();
    let base_bytes: u64 = slices.iter().map(|s| s.base_size).sum();
    eprintln!(
        "mode={} slices={} logs={} base_bytes={} columns={:?}",
        args.mode,
        slices.len(),
        n_logs,
        base_bytes,
        args.columns
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.concurrency.max(1))
        .enable_all()
        .build()
        .unwrap();
    let run_once = || {
        runtime.block_on(async {
            match args.mode.as_str() {
                "hudi" => run_hudi(&args, &slices, true).await,
                "hudi-base" => run_hudi(&args, &slices, false).await,
                "parquet" => run_parquet(&args, &slices).await,
                m => panic!("unknown mode {m}"),
            }
        })
    };
    for it in 0..args.iterations {
        ALLOCS.store(0, Ordering::Relaxed);
        ALLOC_BYTES.store(0, Ordering::Relaxed);
        COUNT_ON.store(args.count_allocs, Ordering::Relaxed);
        let before = rusage::Rusage::capture();
        let start = Instant::now();
        let (rows, batches) = run_once();
        let wall = start.elapsed();
        COUNT_ON.store(false, Ordering::Relaxed);
        let d = rusage::Rusage::capture().delta(&before);
        let cpu_ms = d.user_ms + d.sys_ms;
        println!(
            "RESULT mode={} iter={} rows={} batches={} wall_ms={} cpu_ms={} user_ms={} sys_ms={} cpu_ns_per_row={:.1} mrows_per_cpu_s={:.2} allocs={} alloc_mb={} peak_rss_mb={}",
            args.mode,
            it,
            rows,
            batches,
            wall.as_millis(),
            cpu_ms,
            d.user_ms,
            d.sys_ms,
            cpu_ms as f64 * 1e6 / rows.max(1) as f64,
            rows as f64 / 1e3 / cpu_ms.max(1) as f64,
            ALLOCS.load(Ordering::Relaxed),
            ALLOC_BYTES.load(Ordering::Relaxed) / (1 << 20),
            d.max_rss_kb / 1024
        );
    }
    // Profile separate, untimed iterations so the sampler does not skew the
    // numbers above.
    if let Some(path) = &args.profile {
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(2999)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .unwrap();
        for _ in 0..5 {
            run_once();
        }
        let report = guard.report().build().unwrap();
        let file = std::fs::File::create(path).unwrap();
        report.flamegraph(file).unwrap();
        print_profile(&report, 25);
    }
}
