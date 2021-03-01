#! /usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()
import os
import sys

# ugh
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from aiohttp import web
import output_loader as ol
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryDirectory

# wow this is a shitty way of handling this
# this is a janky in-memory cache
# this works because the webserver is single-threaded
class Cache:
    store = {}
    def make_key(self, file_name, epsilon, key):
        return f"{file_name}_{epsilon}_{key}"
    def has(self, key):
        return key in self.store
    def set(self, key, buf):
        self.store[key] = buf
    def get(self, key):
        return self.store[key] if key in self.store else None

cache = Cache()


def create_bar_chart(cluster_lbls, orig_key):
    # read the data out of the cluster_lbls object into a numpy array
    arr = np.zeros((len(cluster_lbls),2))
    for i, key in enumerate(cluster_lbls):
        arr[i][0] = int(key)
        arr[i][1] = cluster_lbls[key]

    # col 0 is the x values of the bars, col 1 is the heights
    x = arr[:,0]
    heights = arr[:,1]

    # plot the graph to the internal pyplot obj
    plt.close('all')
    plt.bar(x, heights)
    plt.title(f'splits for key {orig_key}')


    # there isn't a nice way of writing a png-encoded buffer out of
    # pyplot, so for simplicity i'm just going to let it hit the disk
    # and read it back
    img = None
    with TemporaryDirectory() as tmpdirname:
        outfile = os.path.join(tmpdirname, 'output.png')
        plt.savefig(
            outfile,
            transparent=True,
            bbox_inches='tight'
        )

        img = open(outfile, 'rb').read()

    # tmpdir should be cleaned up
    return img


async def redir(request):
    raise web.HTTPFound('/index.html')


async def hello(request):
    return web.Response(text="Hello, world")


async def get_available_experiments(request):
    available = os.listdir(os.getenv('CI_EXPERIMENT_DIR'))

    return web.json_response({
        'available': sorted(available),
        'directory': os.getenv('CI_EXPERIMENT_DIR')
    })



async def get_experiment(request):
    print(request.query)
    if not ('name' in request.query) or not ('epsilon' in request.query):
        return web.json_response({
            'ok': False
        })


    run = ol.load_explanations(os.path.join(
        os.getenv('CI_EXPERIMENT_DIR'),
        request.query['name']
    ))

    results = ol.get_variances_labels(
        run,
        float(request.query['epsilon'])
    )


    # convert to python list and histogram
    for key in results:
        results[key]['variances']    = results[key]['variances'].tolist()
        results[key]['cluster_lbls'] = ol.lbl_hist(results[key]['cluster_lbls'])


    del run['shaps']
    del run['y_trn_hw']
    del run['y_tst_hw']
    run['results'] = results
    run['ok'] = True
    run['file_name'] = request.query['name']
    run['epsilon'] = float(request.query['epsilon'])

    for key in run['results']:
        cachekey = cache.make_key(run['file_name'], run['epsilon'], key)
        if not cache.has(cachekey):
            img = create_bar_chart(run['results'][key]['cluster_lbls'], key)
            cache.set(cachekey, img)

    return web.json_response(run)

async def get_chart(request):
    cachekey = cache.make_key(
        request.query['name'],
        float(request.query['epsilon']),
        int(request.query['key'])
    )

    if cache.has(cachekey):
        img = cache.get(cachekey)

        return web.Response(body=img, content_type='image/png')

    else:
        raise web.HTTPNotFound()


if __name__=='__main__':
    app = web.Application()
    app.add_routes([
        web.get('/api/hello', hello),
        web.get('/api/list', get_available_experiments),
        web.get('/api/experiment', get_experiment),
        web.get('/api/chart', get_chart),
        web.get('/', redir),
        web.static('/', 'static')
    ])

    web.run_app(app)
