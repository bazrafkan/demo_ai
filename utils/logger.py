import pandas as pd
from colorful_debug import ColorfulDebug


def debug_log(title, data):
    """
    Display the logs for debugging purposes.

    Args:
        title (str): The title of the log.
        data (list): list of dictionaries containing the logs.
    """
    df_logs = pd.DataFrame(data)
    debug = ColorfulDebug(show_timestamp=True, all_bold=False)
    debug.set_color("info", "blue")
    debug.set_prefix("info", f"[Debug] - {title}")
    
    debug.print("\n", msg_type="info")
    
    debug.set_show_timestamp(False)
    debug.set_prefix("info", "")

    debug.print(df_logs, msg_type="info")
    
    debug.print("\n", msg_type="info")


def error_log(title, data):
    """
    Display the logs for error purposes.

    Args:
        title (str): The title of the log.
        data (list): list of dictionaries containing the logs.
    """
    df_logs = pd.DataFrame(data)
    debug = ColorfulDebug(show_timestamp=True, all_bold=False)
    debug.set_prefix("error", f"[Debug] - {title}")
    
    debug.print("\n", msg_type="error")
    
    debug.set_show_timestamp(False)
    debug.set_prefix("error", "")

    debug.print(df_logs, msg_type="error")
    
    debug.print("\n", msg_type="error")


def save_log(filename, data):
    """
    Save the logs in a CSV file.

    Args:
        title (str): The title of the log.
        data (list): list of dictionaries containing the logs.
        filename (str): The name of the file to save the logs.
    """
    df_logs = pd.DataFrame(data)
    df_logs.to_csv(f"{filename}.csv", index=False)

