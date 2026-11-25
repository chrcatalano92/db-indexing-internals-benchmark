import os
import time
import shutil
import psutil
import numpy as np
import psycopg
import pygtrie
from rocksdict import Rdict, Options, WriteBatch

PG_CONN = (
    f"host={os.getenv('PG_HOST', 'localhost')} "
    f"port={os.getenv('PG_PORT', '5432')} "
    f"dbname={os.getenv('PG_DB', 'indexing_lab')} "
    f"user={os.getenv('PG_USER', 'bench')} "
    f"password={os.getenv('PG_PASSWORD', 'bench_password')}"
)
NUM_RECORDS = int(os.getenv("NUM_RECORDS", 1_000_000))
ROCKS_PATH = "/tmp/rocksdb_bench/lsm_store"


def percentiles_ms(latencies_sec):
    arr = np.array(latencies_sec) * 1000.0
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def run_relational_and_lsm_benchmarks():
    results = []
    rng = np.random.default_rng(42)

    print(f"[1/2] Generating {NUM_RECORDS:,} deterministic records...")
    ids = np.arange(1, NUM_RECORDS + 1, dtype=np.int64)
    skus = [f"CAT-{i % 100:03d}-SKU-{i:08d}" for i in ids]

    n_cluster = int(NUM_RECORDS * 0.8)
    coords_x = np.concatenate([
        rng.normal(40.85, 0.02, n_cluster),
        rng.uniform(40.0, 41.5, NUM_RECORDS - n_cluster),
    ])
    coords_y = np.concatenate([
        rng.normal(14.26, 0.02, n_cluster),
        rng.uniform(13.5, 15.0, NUM_RECORDS - n_cluster),
    ])

    query_indices = rng.choice(NUM_RECORDS, size=2000, replace=False)

    with psycopg.connect(PG_CONN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS catalog_items;")
            cur.execute("""
                CREATE UNLOGGED TABLE catalog_items (
                    id BIGINT NOT NULL,
                    sku TEXT NOT NULL,
                    location POINT NOT NULL
                );
            """)

            t0 = time.perf_counter()
            with cur.copy("COPY catalog_items (id, sku, location) FROM STDIN") as copy:
                for i in range(NUM_RECORDS):
                    copy.write_row((int(ids[i]), skus[i], f"({coords_x[i]},{coords_y[i]})"))
            ingest_sec = time.perf_counter() - t0
            pg_write_tps = NUM_RECORDS / ingest_sec

            # 1. B+ Tree
            t0 = time.perf_counter()
            cur.execute("CREATE INDEX idx_btree_id ON catalog_items USING btree (id);")
            btree_build = time.perf_counter() - t0
            cur.execute("SELECT pg_relation_size('idx_btree_id') / (1024.0 * 1024.0);")
            btree_mb = float(next(iter(cur.fetchone())))

            lats = []
            for idx in query_indices:
                q_id = int(ids[idx])
                ts = time.perf_counter()
                cur.execute("SELECT sku FROM catalog_items WHERE id = %s;", (q_id,))
                cur.fetchone()
                lats.append(time.perf_counter() - ts)
            p50, p95, p99 = percentiles_ms(lats)
            results.append({
                "structure": "B+ Tree (PostgreSQL)",
                "workload": "Point Lookup O(log_B n)",
                "write_ops_sec": round(pg_write_tps, 1),
                "build_time_s": round(btree_build, 2),
                "p50_ms": round(p50, 4),
                "p95_ms": round(p95, 4),
                "p99_ms": round(p99, 4),
                "footprint_mb": round(btree_mb, 2),
            })

            # 2. GiST (Balanced R-Tree)
            t0 = time.perf_counter()
            cur.execute("CREATE INDEX idx_gist_loc ON catalog_items USING gist (location);")
            gist_build = time.perf_counter() - t0
            cur.execute("SELECT pg_relation_size('idx_gist_loc') / (1024.0 * 1024.0);")
            gist_mb = float(next(iter(cur.fetchone())))

            lats = []
            for idx in query_indices[:500]:
                x, y = float(coords_x[idx]), float(coords_y[idx])
                box = f"(({x-0.005},{y-0.005}),({x+0.005},{y+0.005}))"
                ts = time.perf_counter()
                cur.execute("SELECT count(*) FROM catalog_items WHERE location <@ %s::box;", (box,))
                cur.fetchone()
                lats.append(time.perf_counter() - ts)
            p50, p95, p99 = percentiles_ms(lats)
            results.append({
                "structure": "GiST R-Tree (PostgreSQL)",
                "workload": "2D Spatial Range (Skewed)",
                "write_ops_sec": round(pg_write_tps, 1),
                "build_time_s": round(gist_build, 2),
                "p50_ms": round(p50, 4),
                "p95_ms": round(p95, 4),
                "p99_ms": round(p99, 4),
                "footprint_mb": round(gist_mb, 2),
            })

            # 3. SP-GiST (Space-Partitioned QuadTree)
            t0 = time.perf_counter()
            cur.execute("CREATE INDEX idx_spgist_loc ON catalog_items USING spgist (location);")
            spgist_build = time.perf_counter() - t0
            cur.execute("SELECT pg_relation_size('idx_spgist_loc') / (1024.0 * 1024.0);")
            spgist_mb = float(next(iter(cur.fetchone())))

            cur.execute("DROP INDEX idx_gist_loc;")
            lats = []
            for idx in query_indices[:500]:
                x, y = float(coords_x[idx]), float(coords_y[idx])
                box = f"(({x-0.005},{y-0.005}),({x+0.005},{y+0.005}))"
                ts = time.perf_counter()
                cur.execute("SELECT count(*) FROM catalog_items WHERE location <@ %s::box;", (box,))
                cur.fetchone()
                lats.append(time.perf_counter() - ts)
            p50, p95, p99 = percentiles_ms(lats)
            results.append({
                "structure": "SP-GiST QuadTree (PostgreSQL)",
                "workload": "2D Spatial Range (Skewed)",
                "write_ops_sec": round(pg_write_tps, 1),
                "build_time_s": round(spgist_build, 2),
                "p50_ms": round(p50, 4),
                "p95_ms": round(p95, 4),
                "p99_ms": round(p99, 4),
                "footprint_mb": round(spgist_mb, 2),
            })

    # 4. LSM-Tree (RocksDB)
    if os.path.exists(ROCKS_PATH):
        shutil.rmtree(ROCKS_PATH)
    os.makedirs(ROCKS_PATH, exist_ok=True)

    opts = Options(raw_mode=True)
    opts.create_if_missing(True)
    opts.set_write_buffer_size(64 * 1024 * 1024)
    db = Rdict(ROCKS_PATH, options=opts)

    t0 = time.perf_counter()
    batch = WriteBatch(raw_mode=True)
    for i in range(NUM_RECORDS):
        key = int(ids[i]).to_bytes(8, "big")
        val = skus[i].encode("utf-8")
        batch.put(key, val)
        if (i + 1) % 10_000 == 0:
            db.write(batch)
            batch = WriteBatch(raw_mode=True)
    db.write(batch)
    db.flush()
    lsm_write_sec = time.perf_counter() - t0
    lsm_write_tps = NUM_RECORDS / lsm_write_sec

    lats = []
    for idx in query_indices:
        q_key = int(ids[idx]).to_bytes(8, "big")
        ts = time.perf_counter()
        _ = db[q_key]
        lats.append(time.perf_counter() - ts)
    p50, p95, p99 = percentiles_ms(lats)
    db.close()
    lsm_mb = get_dir_size_mb(ROCKS_PATH)

    results.append({
        "structure": "LSM-Tree (RocksDB)",
        "workload": "Write-Heavy + Point Lookup",
        "write_ops_sec": round(lsm_write_tps, 1),
        "build_time_s": round(lsm_write_sec, 2),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "footprint_mb": round(lsm_mb, 2),
    })

    # 5. Radix Tree (Compact Trie)
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)
    trie = pygtrie.CharTrie()

    t0 = time.perf_counter()
    trie_limit = min(NUM_RECORDS, 300_000)
    for i in range(trie_limit):
        trie[skus[i]] = int(ids[i])
    trie_build = time.perf_counter() - t0
    mem_after = proc.memory_info().rss / (1024 * 1024)

    lats = []
    for idx in query_indices:
        prefix = skus[idx % trie_limit][:15]
        ts = time.perf_counter()
        _ = list(trie.iterkeys(prefix=prefix))[:10]
        lats.append(time.perf_counter() - ts)
    p50, p95, p99 = percentiles_ms(lats)

    results.append({
        "structure": "Radix Tree (Compact Trie)",
        "workload": "Prefix Search O(k)",
        "write_ops_sec": round(trie_limit / trie_build, 1),
        "build_time_s": round(trie_build, 2),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "footprint_mb": round(max(mem_after - mem_before, 1.0), 2),
    })

    return results