import dask.dataframe as dd


data = dd.read_csv("sample.csv", dtype={"number": "object"})
print(data.head())


sum_data = data.lower + data.upper
print(sum_data)

result = sum_data.compute()
print(result.head())


means = data[["lower", "upper"]].mean().compute()
print(means)
