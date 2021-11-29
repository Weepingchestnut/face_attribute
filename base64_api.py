import json
import platform

import threadpool
import tornado.httpserver
import tornado.ioloop
import tornado.process
import tornado.web

from interface_main import base64_api
from mylogger import logger


def make_app(config):
    threadpools = threadpool.ThreadPool(config.get("maxThread", 200))
    config["threadpool"] = threadpools
    return tornado.web.Application([
        (r"/lzk/base64", base64_api, dict(gconf=config))
    ])


def main():
    with open("port.json", "r") as fp:
        config = json.load(fp)

    systemType = platform.system()  # "windows"#

    logger.debug("platform system type is %s" % systemType)
    port = config.get("face_attr", 8704)
    logger.debug("start listening", port)

    if systemType.lower() == "windows":
        logger.debug("start single process")
        app = make_app(config)
        app.listen(port)
        logger.info("face_attribute process is started")
        tornado.ioloop.IOLoop.current().start()

    else:
        logger.debug("start multi processes")
        sockets = tornado.netutil.bind_sockets(port)
        # tornado.process.fork_processes(1)
        app = make_app(config)
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.add_sockets(sockets)
        logger.info("face_attribute process is started")
        tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
