
# safegraph-simulation

ASSIP 2020 research project to simulate the spread of COVID-19 by applying LDA to data from SafeGraph

## Installation

1. Clone the repository.

2. Create a subdirectory called `safegraph-data` in the repository directory.

3. Download the SafeGraph Open Census Data [here](https://www.safegraph.com/open-census-data). Save it to `safegraph-data/safegraph_open_census_data`.

4. Download the SafeGraph Weekly Patterns (v2) data after following [these steps](https://www.safegraph.com/covid-19-data-consortium). Save it to `safegraph-data/safegraph_weekly_patterns_v2`.

5. Unzip each .gz file in `safegraph-data/safegraph_weekly_patterns_v2/main-file` as necessary. Save each unzipped .csv file to `safegraph-data/safegraph_weekly_patterns_v2/main-file/csv_name_without_extension/csv_name_without_extension.csv`.
Example: `safegraph-data/safegraph_weekly_patterns_v2/main-file/2020-06-01-weekly-patterns.csv.gz` unzips to `safegraph-data/safegraph_weekly_patterns_v2/main-file/2020-06-01-weekly-patterns/2020-06-01-weekly-patterns.csv`.

6. Download the SafeGraph Social Distancing Metrics data after following the same process outlined in step 4. Save it to `safegraph-data/safegraph_social_distancing_metrics`. Unzip each .gz file as necessary.
Example: `safegraph-data/safegraph_social_distancing_metrics/2020/06/01/2020-06-01-social-distancing.csv.gz` unzips to `safegraph-data/safegraph_social_distancing_metrics/2020/06/01/2020-06-01-social-distancing.csv`.

## Usage

`county_parser.py` is used to extract locality data from the main .csv file. Run this at least once, and then run `simulation.py`.