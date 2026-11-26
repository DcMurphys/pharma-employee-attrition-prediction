"""Batch scoring controller (Check an Employee, tab 2; scope §4.3).

Uploads stay in memory for the current session only. Validation, ranking and flagging reuse
``validate_records`` and ``score_batch``.
"""

import logging

import pandas as pd
import streamlit as st

from dashboard.controllers import data
from dashboard.models import batch, export
from dashboard.models.scoring import Issue, score_batch, validate_records
from dashboard.views import batch as view
from dashboard.views.components import notices
from dashboard.views.components.download import csv_download
from employee_attrition.config import DATA_DISCLAIMER, REVIEW_CAPACITY_SHARE

logger = logging.getLogger(__name__)

KEY_SOURCE = "batch_source"
KEY_UPLOAD = "batch_upload"
KEY_SAMPLE = "batch_sample"
SOURCE_UPLOAD = "Upload a CSV file"
SOURCE_SAMPLE = "Use a sample file"


def render() -> None:
    """Draw the batch scoring tab."""
    view.render_intro()
    schema, features = data.feature_schema(), data.features()
    samples = batch.sample_frames(data.workforce(), data.risk_scores(), features)
    _sample_downloads(samples)

    source = st.radio(
        "Data to score", [SOURCE_UPLOAD, SOURCE_SAMPLE], key=KEY_SOURCE, horizontal=True
    )
    records = _records(source, samples)
    if records is None:
        return

    issues = validate_records(records, schema)
    view.render_issues(
        [_issue_text(i, schema) for i in issues if i.blocking],
        [_issue_text(i, schema) for i in issues if not i.blocking],
    )
    if any(issue.blocking for issue in issues):
        return

    scored = score_batch(data.model(), batch.prepare(records, schema), features)
    result = batch.summary(scored)
    view.render_summary(result.rows, result.flagged, result.mean_probability, REVIEW_CAPACITY_SHARE)
    if result.rows < batch.MIN_MEANINGFUL_ROWS:
        view.render_small_batch_note(batch.MIN_MEANINGFUL_ROWS)

    id_label = "Employee ID" if records.index.name != batch.ROW_LABEL else batch.ROW_LABEL
    table = view.results_table(scored, REVIEW_CAPACITY_SHARE, id_label)
    view.render_results(table)
    download = pd.concat([table, scored[features].reset_index(drop=True)], axis=1)
    csv_download(
        "Download the scored file (CSV)",
        export.csv_with_notes(download, [DATA_DISCLAIMER, notices.INTENDED_USE]),
        export.demo_file_name("batch_scores"),
        key="batch_download",
    )


def _records(source: str, samples: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Records from the upload or the chosen sample; ``None`` until there is something to score."""
    if source == SOURCE_SAMPLE:
        name = st.selectbox(
            "Sample file",
            list(batch.SAMPLES),
            key=KEY_SAMPLE,
            format_func=batch.SAMPLES.get,
            width=360,
        )
        return batch.read_csv_bytes(_sample_bytes(samples[name]))
    upload = st.file_uploader(
        "CSV file (up to 5 MB)", type="csv", key=KEY_UPLOAD, help=view.UPLOAD_HINT
    )
    if upload is None:
        return None
    try:
        return batch.read_csv_bytes(upload.getvalue())
    except batch.BatchFileError as err:
        logger.warning("Upload rejected: %s", err)
        view.render_file_error(err.kind)
        return None


def _sample_bytes(frame: pd.DataFrame) -> bytes:
    """A sample exactly as it is downloaded, disclaimer included."""
    return export.csv_with_notes(frame, [DATA_DISCLAIMER])


def _sample_downloads(samples: dict[str, pd.DataFrame]) -> None:
    """The three demo files, in an expander."""
    with st.expander("Sample files", icon=":material/description:"):
        st.caption(
            "Demo files built from employees awaiting an outcome. Download one to see the "
            "expected format."
        )
        for name, frame in samples.items():
            csv_download(
                batch.SAMPLES[name],
                _sample_bytes(frame),
                export.demo_file_name(f"sample_{name}"),
                key=f"batch_sample_download_{name}",
            )


def _issue_text(issue: Issue, schema: list[dict]) -> view.IssueText:
    """Resolve an ``Issue`` to the values its message needs (scale, range or categories)."""
    field = next((f for f in schema if f["feature"] == issue.column), {})
    return view.IssueText(
        column=issue.column,
        kind=issue.kind,
        rows=issue.n_rows,
        low=field.get("min"),
        high=field.get("max"),
        allowed=", ".join(field.get("categories", "").split("|"))
        if field.get("categories")
        else "",
    )
