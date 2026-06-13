import re
import pandas as pd

def preprocess(data):

    # WhatsApp pattern
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} (?:AM|PM) - '

    # Split messages
    messages = re.split(pattern, data)[1:]

    # Extract dates
    dates = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame({
        'user_message': messages,
        'message_date': dates
    })

    # Convert date
    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format='%d/%m/%y, %I:%M %p - '
    )

    df.rename(columns={
        'message_date': 'date'
    }, inplace=True)

    users = []
    messages_list = []

    for message in df['user_message']:

        entry = re.split(r'([\w\W]+?):\s', message)

        if entry[1:]:
            users.append(entry[1])
            messages_list.append(entry[2])

        else:
            users.append('group_notification')
            messages_list.append(entry[0])

    df['user'] = users
    df['message'] = messages_list

    # Remove old column
    df.drop(columns=['user_message'], inplace=True)

    # Date features
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Only date
    df['only_date'] = df['date'].dt.date

    # Day name
    df['day_name'] = df['date'].dt.day_name()

    return df