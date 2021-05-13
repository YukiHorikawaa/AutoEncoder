import concat
Concat = concat.concat("Obrid_AE", "data")
Concat.output_data("sample_data")
Concat = concat.concat("Obrid_AE", "test")
Concat.output_data("sample_test")
print("----------------------")
print(Concat.get_data("sample_test").shape)