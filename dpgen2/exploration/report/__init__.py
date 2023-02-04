from .report import (
    ExplorationReport,
)
from .report_adaptive_lower import (
    ExplorationReportAdaptiveLower,
)
from .report_trust_levels import (
    ExplorationReportTrustLevels,
)

conv_styles = {
    "fixed-levels": ExplorationReportTrustLevels,
    "adaptive-lower": ExplorationReportAdaptiveLower,
}
