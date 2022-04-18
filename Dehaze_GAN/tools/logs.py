import os
import datetime


def write_infor(root, filename='result', category=None, infor=None):
    file_path = os.path.join(root, filename)
    with open(file_path, mode='a', encoding='utf-8') as f:
        if category == 'start_time' or category == 'end_time':
            f.write('{} : {}\n'.format(
                category,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        elif category == 'duration':
            duration = str(datetime.timedelta(seconds=int(infor)))
            f.write(category + ' : ' + duration + '\n')
        else:
            f.write(category.ljust(10) + ':' + str(infor) + '\n')
