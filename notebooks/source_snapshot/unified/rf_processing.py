"""RF deterministic processing, retained from original cells 2/41/42/44."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def prepare_model_data(input_data):
    """
    将Part 1输出的数据转换为统一的模型输入。

    只进行必要的确定性处理：
    1. 修正INSTALL字段名称；
    2. 将无穷值转为缺失值；
    3. 将年龄和就业天数转换为年；
    4. 处理DAYS_EMPLOYED特殊编码365243；
    5. 添加历史表无记录和缺失标记；
    6. 删除无意义的索引列。

    不在这里进行中位数填补和One-Hot，
    这些步骤必须只在训练集上拟合。
    """

    model_data = input_data.copy()

    # --------------------------------------------------------
    # 1. INSTALL字段名称修正
    # --------------------------------------------------------

    old_install_name = "INSTALL_AVG_DAYS_LATE"
    new_install_name = "INSTALL_AVG_DAYS_ENTRY_PAYMENT"

    if (
        old_install_name in model_data.columns
        and new_install_name not in model_data.columns
    ):
        model_data = model_data.rename(
            columns={
                old_install_name: new_install_name
            }
        )

    if (
        old_install_name in model_data.columns
        and new_install_name in model_data.columns
    ):
        raise ValueError(
            "INSTALL新旧字段同时存在，请检查数据版本。"
        )

    # --------------------------------------------------------
    # 2. 无穷值转为缺失值，后续交给训练集填补器
    # --------------------------------------------------------

    numeric_columns = model_data.select_dtypes(
        include=["number", "bool"]
    ).columns

    model_data[numeric_columns] = (
        model_data[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # 3. 年龄转换为年
    # --------------------------------------------------------

    if "DAYS_BIRTH" in model_data.columns:
        model_data["YEARS_BIRTH"] = (
            -model_data["DAYS_BIRTH"] / 365.25
        )

        model_data = model_data.drop(
            columns=["DAYS_BIRTH"]
        )

    # --------------------------------------------------------
    # 4. DAYS_EMPLOYED特殊编码处理
    # --------------------------------------------------------

    if "DAYS_EMPLOYED" in model_data.columns:
        employment_sentinel = (
            model_data["DAYS_EMPLOYED"] == 365243
        )

        model_data["EMPLOYMENT_SENTINEL_FLAG"] = (
            employment_sentinel.astype("int8")
        )

        model_data["YEARS_EMPLOYED"] = (
            -model_data["DAYS_EMPLOYED"]
            .mask(employment_sentinel)
            / 365.25
        )

        model_data = model_data.drop(
            columns=["DAYS_EMPLOYED"]
        )

    # --------------------------------------------------------
    # 5. 历史表无记录和缺失标记
    # --------------------------------------------------------

    history_groups = {
        "BUREAU": {
            "prefix": "BUREAU_",
            "count_columns": [
                "BUREAU_COUNT",
                "BUREAU_ACTIVE",
                "BUREAU_CLOSED"
            ]
        },
        "PREV": {
            "prefix": "PREV_",
            "count_columns": [
                "PREV_COUNT",
                "PREV_APPROVED",
                "PREV_REFUSED"
            ]
        },
        "INSTALL": {
            "prefix": "INSTALL_",
            "count_columns": [
                "INSTALL_COUNT"
            ]
        },
        "POS": {
            "prefix": "POS_",
            "count_columns": [
                "POS_COUNT"
            ]
        },
        "CC": {
            "prefix": "CC_",
            "count_columns": [
                "CC_COUNT"
            ]
        }
    }

    for group_name, group_config in history_groups.items():

        no_record_column = f"{group_name}_NO_RECORD"

        # 排除已经生成的缺失标记，防止重复运行时受到影响
        history_columns = [
            column
            for column in model_data.columns
            if column.startswith(group_config["prefix"])
            and column != no_record_column
            and not column.endswith("_MISSING")
        ]

        if not history_columns:
            continue

        no_record_mask = (
            model_data[history_columns]
            .isna()
            .all(axis=1)
        )

        model_data[no_record_column] = (
            no_record_mask.astype("int8")
        )

        # 只有在整张历史表没有记录时，计数字段才填0
        for count_column in group_config["count_columns"]:
            if count_column in model_data.columns:
                model_data.loc[
                    no_record_mask,
                    count_column
                ] = 0

        # 其他历史变量保留缺失标记
        for column in history_columns:
            if (
                column not in group_config["count_columns"]
                
            ):
                missing_flag = f"{column}_MISSING"

                model_data[missing_flag] = (
                    model_data[column]
                    .isna()
                    .astype("int8")
                )

    # --------------------------------------------------------
    # 6. 删除CSV保存时可能产生的索引列
    # --------------------------------------------------------

    index_columns = [
        column
        for column in model_data.columns
        if column.lower().startswith("unnamed:")
    ]

    if index_columns:
        model_data = model_data.drop(
            columns=index_columns
        )

    for column in input_data.columns:
        if column.startswith(('R_', 'H_', 'BB_')) and not column.endswith('_MISSING'):
            model_data[f'{column}_MISSING'] = model_data[column].isna().astype('int8')
    return model_data

repeated_property_bases = [
    "APARTMENTS",
    "BASEMENTAREA",
    "YEARS_BEGINEXPLUATATION",
    "YEARS_BUILD",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA"
]
business_ratio_definitions = {
    "CREDIT_ANNUITY_RATIO": (
        "AMT_CREDIT",
        "AMT_ANNUITY"
    ),

    "CREDIT_GOODS_RATIO": (
        "AMT_CREDIT",
        "AMT_GOODS_PRICE"
    ),

    "BUREAU_DEBT_CREDIT_RATIO": (
        "BUREAU_AVG_DEBT",
        "BUREAU_AVG_CREDIT"
    ),

    "BUREAU_ACTIVE_RATE": (
        "BUREAU_ACTIVE",
        "BUREAU_COUNT"
    ),

    "BUREAU_CLOSED_RATE": (
        "BUREAU_CLOSED",
        "BUREAU_COUNT"
    ),

    "PREV_APPROVAL_RATE": (
        "PREV_APPROVED",
        "PREV_COUNT"
    ),

    "PREV_REFUSAL_RATE": (
        "PREV_REFUSED",
        "PREV_COUNT"
    ),

    "CC_BALANCE_LIMIT_RATIO": (
        "CC_AVG_BALANCE",
        "CC_AVG_LIMIT"
    )
}

def safe_ratio(
    dataframe,
    numerator_column,
    denominator_column
):
    """
    安全计算比率：
    1. 分母为0时转为缺失；
    2. 正负无穷转为缺失；
    3. 后续由训练折中位数填补。
    """

    if numerator_column not in dataframe.columns:
        raise KeyError(
            f"缺少分子字段：{numerator_column}"
        )

    if denominator_column not in dataframe.columns:
        raise KeyError(
            f"缺少分母字段：{denominator_column}"
        )

    denominator_values = (
        dataframe[denominator_column]
        .replace(0, np.nan)
    )

    ratio_values = (
        dataframe[numerator_column]
        / denominator_values
    )

    ratio_values = ratio_values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return ratio_values

def build_feature_preprocessor(
    feature_dataframe
):
    """
    针对当前特征集合构建预处理器。
    每折只在训练部分拟合。
    """

    numeric_columns = (
        feature_dataframe
        .select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        feature_dataframe
        .select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    recognized_columns = (
        numeric_columns
        + categorical_columns
    )

    unknown_columns = [
        column
        for column in feature_dataframe.columns
        if column not in recognized_columns
    ]

    if unknown_columns:
        raise TypeError(
            f"存在无法识别的数据类型字段："
            f"{unknown_columns}"
        )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(
                    strategy="median", keep_empty_features=True
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "missing_imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing"
                )
            ),
            (
                "one_hot_encoder",
                encoder
            )
        ]
    )

    feature_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True
    )

    return (
        feature_preprocessor,
        numeric_columns,
        categorical_columns
    )

def f3_features(frame):
    x = prepare_model_data(frame)
    x = x.drop(columns=[c for c in x if c in ['TARGET','partition','fold'] or c.startswith('SK_ID')])
    redundant = {f'{base}_{suffix}' for base in repeated_property_bases for suffix in ['MODE','MEDI']}
    result = x[[c for c in x if c not in redundant]].copy()
    for name, (numerator, denominator) in business_ratio_definitions.items():
        result[name] = safe_ratio(x, numerator, denominator)
    return result
