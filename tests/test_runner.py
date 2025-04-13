from project_validator import test_essential_functionality, collect_core_metrics

# Run the main test
test_essential_functionality()

# Collect and print metrics
metrics = collect_core_metrics()
print("Test metrics:")
for key, value in metrics.items():
    print(f"- {key}: {value}")