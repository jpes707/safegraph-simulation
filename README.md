
# safegraph-simulation

ASSIP 2020 research project to simulate the spread of COVID-19 by applying LDA to data from SafeGraph

## Installation

1. Clone the repository.

2. Create a subdirectory called `safegraph-data` in the repository directory.

3. Download the SafeGraph Open Census Data [here](https://www.safegraph.com/open-census-data). Save it to `safegraph-data/safegraph_open_census_data`.

4. Download the SafeGraph Weekly Patterns (v2) data after following [these steps](https://www.safegraph.com/covid-19-data-consortium). Save it to `safegraph-data/safegraph_weekly_patterns_v2`.

5. Unzip all .gz files in `safegraph-data/safegraph_weekly_patterns_v2/main-file`. Save each unzipped .csv file to `safegraph-data/safegraph_weekly_patterns_v2/main-file/csv_name_without_extension/csv_name_without_extension.csv`.
Example: `safegraph-data/safegraph_weekly_patterns_v2/main-file/2020-06-01-weekly-patterns.csv.gz` unzips to `safegraph-data/safegraph_weekly_patterns_v2/main-file/2020-06-01-weekly-patterns/2020-06-01-weekly-patterns.csv`.

## Usage

`county_parser.py` and `state_parser.py` are used to extract locality data from the main .csv file. Run one of these files at least once, and then run `lda.py`.