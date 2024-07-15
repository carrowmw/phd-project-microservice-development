from dashboard.data import CustomDashboardData

training_windows = CustomDashboardData().get_training_windows()

if __name__ == "__main__":
    print(training_windows)
