from numaprom.monitoring.utility import get_metric


def test_get_metric():
    name = "test_counter"
    description = "test counter"

    name_info = "test_info"
    description_info = "test info"

    name_summary = "test_summary"
    description_summary = "test summary"

    name_gauge = "test_gauge"
    description_gauge = "test gauge"

    label_pairs = {"label1": "value1"}
    metric = get_metric("Counter", name, description, label_pairs)
    metric_info = get_metric("Info", name_info, description_info, label_pairs)
    metric_summary = get_metric("Summary", name_summary, description_summary, label_pairs)
    metric_gauge = get_metric("Gauge", name_gauge, description_gauge, label_pairs)

    assert metric.name == name
    assert metric.description == description
    assert metric_info.name == name_info
    assert metric_info.description == description_info
    assert metric_summary.name == name_summary
    assert metric_summary.description == description_summary
    assert metric_gauge.name == name_gauge
    assert metric_gauge.description == description_gauge
