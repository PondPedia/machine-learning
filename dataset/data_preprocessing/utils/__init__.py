def get_user_inputs():
    """
    Get user inputs for data cleaning or data sampling.

    Returns:
        tuple: A tuple containing options dictionary, user's choice, pond number,
               split amount and interval.
    """

    # Define options dictionary
    options = {1: "Cleaning", 2: "Sampling"}

    # Print options dictionary
    for key, value in options.items():
        print("{}: {}".format(key, value))

    # Get user's choice
    choice = int(input("Enter your choice: "))

    # Get pond number
    pond = int(input("Pond: "))

    # Get split amount
    split_amount = int(input("Split Amount: "))

    # Get interval
    interval = input("Interval: ")

    # Return a tuple containing options dictionary, user's choice, pond number,
    # split amount and interval
    return options, choice, pond, split_amount, interval
