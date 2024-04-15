---
    layout: ../../layouts/blogpost.astro
    title: Polars - The Better Pandas?
    pubDate: 2024-04-09
    description: "Polars is an alternative to Pandas with a Rust backend"
    tags: ["Polars", "Pandas"]
---

## Polars - The Better Pandas?

(Draft)

## Contents

## Introduction

Pandas is one of Python's most used data manipulating libraries. It's used widely but it has the problem that as soon as the data grows too big, it's just too slow and companies use other solutions for data processing. Polars on the other hand is written in Rust and has an excellent Python library. In this blog post, we will explore the differences between the two, what kind of performance you can expect with either of them and which I recommend for what situations.

## Performance

Let's start with what most people are interested in: performance. For that we will test a couple of scenarios and compare their speeds. We'll keep a running tab on the score. Let's first create a dummy dataset with 10,000,000 rows and 21 columns (the first one being the `ID`).

```python
import polars as pl
import pandas as pd
import numpy as np
import time
from loguru import logger
from functools import partial
import csv
import os

DATA_PATH = "data.csv"

def log_time(pl_time, pd_time):
logger.info(
f"Polars was {pd_time / pl_time}x {'faster' if pl_time < pd_time else 'slower'} than Pandas"
)

def create_csv():
n_rows = 10_000_000
n_cols = 20

ids = np.arange(n_rows, dtype="int")
cols = np.random.uniform(size=(n_rows, n_cols))

data = np.c_[ids, cols]
np.savetxt(DATA_PATH, data, delimiter=",")

def add_header_to_csv(input_filename, output_filename, headers):
    with open(input_filename, "r", newline="") as infile, open(
        output_filename, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(headers)
        for row in reader:
            writer.writerow(row)

headers = ["ID"] + [f"Col_{i}" for i in range(1, 21)]
add_header_to_csv(DATA_PATH, "data_with_headers.csv", headers)
os.remove(DATA_PATH)
os.rename("data_with_headers.csv", DATA_PATH)

```

This will create a ~5 GB file, so make sure you have enough space.

### Read CSV

Now it's time to test the performance. First, let's check how long it takes to run `px.read_csv` (`px` is the placeholder for when I'm referring to both Pandas and Polars).

```python
def test_open_csv():
    pl_start_time = time.time()
    pl.read_csv(DATA_PATH)
    pl_end_time = time.time()
    pl_time = pl_end_time - pl_start_time
    logger.info(f"Polars: {pl_time}")

    pd_start_time = time.time()
    pd.read_csv(DATA_PATH)
    pd_end_time = time.time()
    pd_time = pd_end_time - pd_start_time
    logger.info(f"Pandas: {pd_time}")

    log_time(pl_time, pd_time)
```

| Task     | Polars | Pandas  |
| -------- | ------ | ------- |
| read_csv | 2.926s | 21.093s |

Polars was 7.2077x faster than Pandas
