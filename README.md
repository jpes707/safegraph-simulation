
# safegraph-simulation

ASSIP 2020 research project to simulate the spread of COVID-19 by applying LDA to data from SafeGraph

## Installation

1. Clone the repository.

2. Create a subdirectory called `safegraph-data` in the repository directory.

3. Download the SafeGraph Open Census Data [here](https://www.safegraph.com/open-census-data). Save it to `safegraph-data/safegraph_open_census_data`.

4. Download the SafeGraph Weekly Patterns (v2) data after following [these steps](https://www.safegraph.com/covid-19-data-consortium). Save it to `safegraph-data/safegraph_weekly_patterns_v2`.

5. Download the SafeGraph Core Places data after following the same process outlined in step 4. Save it to `safegraph-data/safegraph_core_places`.

6. Download the SafeGraph Social Distancing Metrics data after following the same process outlined in step 4. Save it to `safegraph-data/safegraph_social_distancing_metrics`.

## Usage

`county_parser.py` is used to extract locality data. Run this at least once, and then run `simulation.py`.
