from raise_utils.data import Data, DataLoader


timelines = ["1 day", "7 days", "14 days", "30 days", "90 days"]
datasets = ["camel", "cloudstack", "cocoon", "hadoop", "deeplearning", "hive", "node", "ofbiz", "qpid"]


def github_issue_loader():
    for timeline in timelines:
        for dataset in datasets:
            yield load_github_issue_data("../data/issue_close_time_prediction/github/", dataset, timeline)


def load_github_issue_data(base_path: str, dataset: str, timeline: str) -> tuple[str, Data]:
    return f"github:{dataset}-{timeline.replace(' ', '_')}", DataLoader.from_file(
        f"{base_path}/{timeline}/{dataset}.csv", target="timeOpen", col_start=0
    )
