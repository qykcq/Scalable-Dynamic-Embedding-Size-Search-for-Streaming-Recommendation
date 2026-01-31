from Engine import Engine


if __name__ == '__main__':
    for ID in [0, 1]:
        print('########################### ID = {} ###########################'.format(ID))
        engine = Engine(ID)

        engine.streaming_size_search()
