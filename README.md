
# safegraph-simulation

SIGSPATIAL ARIC 2020: Foot-Traffic Informed COVID-19 Simulation and Mitigation

## Installation

1. Clone the repository.

2. Create a subdirectory called `safegraph-data` in the repository directory.

3. Download the SafeGraph Open Census Data [here](https://www.safegraph.com/open-census-data). Save it to `safegraph-data/safegraph_open_census_data`.

4. Get free access to the remaining SafeGraph data by following [these steps](https://www.safegraph.com/covid-19-data-consortium).

4. Download the SafeGraph Weekly Patterns (v2) data. Save it to `safegraph-data/safegraph_weekly_patterns_v2`.

5. Download the SafeGraph Core Places data. Save it to `safegraph-data/safegraph_core_places`.

6. Download the SafeGraph Social Distancing Metrics data. Save it to `safegraph-data/safegraph_social_distancing_metrics`.

## Usage

`county_parser.py` is used to extract locality data. Run this at least once, and then run `simulation.py`.
