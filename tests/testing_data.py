import pandas as pd
import great_expectations as ge
from great_expectations.dataset import PandasDataset

projects = pd.read_csv("data\Selfie_reduced\processed\selfie_dataset.csv")
#tags = pd.read_csv("data/tags.csv")
df = ge.dataset.PandasDataset(projects)
print(df.head(5))


# Presence of specific features
df.expect_table_columns_to_match_ordered_list(column_list=["image_name","popularity_score","partial_faces", "is_female", "baby", "child", "teenager", "youth", "middle_age", "senior",
            "white", "black", "asian", "oval_face", "round_face", "heart_face", "smiling", "mouth_open", "frowning", "wearing_glasses", "wearing_sunglasses", "wearing_lipstick", 
            "tongue_out", "duck_face", "black_hair", "blond_hair", "brown_hair", "red_hair", "curly_hair", "straight_hair", "braid_hair", "showing_cellphone", "using_earphone",
            "using_mirror", "braces", "wearing_hat", "harsh_lighting", "dim_lighting"
])


# Missing values
df.expect_column_values_to_not_be_null(
    column="image_name", 
    column="popularity_score",
    column="partial_faces", 
    column="is_female", 
    column="baby", 
    column="child", 
    column="teenager", 
    column="youth", 
    column="middle_age", 
    column="senior", 
    column= "white", 
    column="black", 
    column="asian", 
    column="oval_face", 
    column="round_face", 
    column="heart_face", 
    column="smiling", 
    column="mouth_open", 
    column="frowning", 
    column="wearing_glasses", 
    column="wearing_sunglasses", 
    column="wearing_lipstick", 
    column="tongue_out", 
    column="duck_face",
    column="black_hair",
    column="blond_hair", 
    column="brown_hair", 
    column="red_hair",
    column="curly_hair", 
    column="straight_hair", 
    column="braid_hair", 
    column="showing_cellphone", 
    column="using_earphone", 
    column="using_mirror", 
    column="braces", 
    column="wearing_hat", 
    column="harsh_lighting", 
    column="dim_lighting")

# Unique values
df.expect_column_values_to_be_unique(column="image_name")


# Type adherence
df.expect_column_values_to_be_of_type(column="image_name", type_="str")
df.expect_column_values_to_be_of_type(column="popularity_score", type_="float")
df.expect_column_values_to_be_of_type(column="partial_faces", type_="int")
df.expect_column_values_to_be_of_type(column="is_female", type_="int")
df.expect_column_values_to_be_of_type(column="baby", type_="int")
df.expect_column_values_to_be_of_type(column="child", type_="int")
df.expect_column_values_to_be_of_type(column="teenager", type_="int")
df.expect_column_values_to_be_of_type(column="youth", type_="int")
df.expect_column_values_to_be_of_type(column="middle_age", type_="int")
df.expect_column_values_to_be_of_type(column="senior", type_="int")
df.expect_column_values_to_be_of_type(column= "white", type_="int")
df.expect_column_values_to_be_of_type(column="black", type_="int")
df.expect_column_values_to_be_of_type(column="asian", type_="int")
df.expect_column_values_to_be_of_type(column="oval_face", type_="int")
df.expect_column_values_to_be_of_type(column="round_face", type_="int")
df.expect_column_values_to_be_of_type(column="heart_face", type_="int")
df.expect_column_values_to_be_of_type(column="smiling", type_="int")
df.expect_column_values_to_be_of_type(column="mouth_open", type_="int")
df.expect_column_values_to_be_of_type(column="frowning", type_="int")
df.expect_column_values_to_be_of_type(column="wearing_glasses", type_="int")
df.expect_column_values_to_be_of_type(column="wearing_sunglasses", type_="int")
df.expect_column_values_to_be_of_type(column="wearing_lipstick", type_="int")
df.expect_column_values_to_be_of_type(column="tongue_out", type_="int")
df.expect_column_values_to_be_of_type(column="duck_face", type_="int")
df.expect_column_values_to_be_of_type(column="black_hair", type_="int")
df.expect_column_values_to_be_of_type(column="blond_hair", type_="int")
df.expect_column_values_to_be_of_type(column="brown_hair", type_="int")
df.expect_column_values_to_be_of_type(column="red_hair", type_="int")
df.expect_column_values_to_be_of_type(column="curly_hair", type_="int")
df.expect_column_values_to_be_of_type(column="straight_hair", type_="int")
df.expect_column_values_to_be_of_type(column="braid_hair", type_="int")
df.expect_column_values_to_be_of_type(column="showing_cellphone", type_="int")
df.expect_column_values_to_be_of_type(column="using_earphone", type_="int")
df.expect_column_values_to_be_of_type(column="using_mirror", type_="int")
df.expect_column_values_to_be_of_type(column="braces", type_="int")
df.expect_column_values_to_be_of_type(column="wearing_hat", type_="int")
df.expect_column_values_to_be_of_type(column="harsh_lighting", type_="int")
df.expect_column_values_to_be_of_type(column="dim_lighting", type_="int")


# List (categorical) or range (continuous)
# of allowed values

df.expect_column_values_to_be_between(column="popularity_score", min_value=0, max_value=10)

binary_values = [1, -1]
df.expect_column_values_to_be_in_set(column="partial_faces", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="is_female", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="baby", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="child", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="teenager", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="youth", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="middle_age", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="senior", value_set=binary_values)
df.expect_column_values_to_be_in_set(column= "white", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="black", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="asian", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="oval_face", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="round_face", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="heart_face", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="smiling", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="mouth_open", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="frowning", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="wearing_glasses", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="wearing_sunglasses", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="wearing_lipstick", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="tongue_out", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="duck_face", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="black_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="blond_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="brown_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="red_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="curly_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="straight_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="braid_hair", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="showing_cellphone", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="using_earphone", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="using_mirror", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="braces", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="wearing_hat", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="harsh_lighting", value_set=binary_values)
df.expect_column_values_to_be_in_set(column="dim_lighting", value_set=binary_values)




# Expectation suite
expectation_suite = df.get_expectation_suite(
discard_failed_expectations=False
)
print(
df.validate(
expectation_suite=expectation_suite,
only_return_failures=True
)
)

