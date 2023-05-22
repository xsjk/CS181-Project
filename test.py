import asyncio

def self_add(i):
    return i+1

async def test(i):
    print('test_1', i)
    a = 0
    await asyncio.sleep(1)
    print('test_2', i)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    tasks = [test(i) for i in range(3)]
    loop.run_until_complete(asyncio.wait(tasks))
    print(11111)
    loop.close()