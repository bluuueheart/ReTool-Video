"""Merged public capabilities; legacy tools remain executable adapters."""

MERGED_TOOL_NAMES = {
    "Boolean_And": ("All_True",),
    "Boolean_Or": ("Any_True",),
    "Percentile_50": ("Median_Field",),
    "Keyword_Filter": ("Filter_Contains",),
    "Score_Threshold_Rank": ("Threshold_Filter",),
    "TopK_By_Score": ("TopK_By_Field",),
    "BottomK_By_Score": ("BottomK_By_Field",),
    "Merge_Touching_Ranges": ("Merge_Temporal_Segments",),
    "Aggregation": ("Count_Items", "Sort_Items", "Max_Item", "Min_Item", "Merge_Temporal_Segments"),
}


def legacy_names_for(name):
    return [alias for alias, replacements in MERGED_TOOL_NAMES.items() if name in replacements]
