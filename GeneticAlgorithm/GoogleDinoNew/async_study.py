# python异步学习
import asyncio

times = 0

async def hello():
    global times
    times += 1
    print("Hello world! %d" % times)
    r = await asyncio.sleep(1)
    print("Hello again! %d" % times)




# 获取EventLoop:\
loop = asyncio.get_event_loop()
# 执行coroutine
loop.run_until_complete(asyncio.wait([hello(),hello(),hello(),hello(),hello(),hello()]))
loop.close()
