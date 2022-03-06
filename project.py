#!/usr/bin/env python3

import sys
import os
import os.path
import urllib.request
from hashlib import sha256
import json
import glob
import importlib

def file_hash(filename):
    with open(filename, 'rb') as f:
        payload = f.read()
    return sha256(payload).hexdigest()
    
class Main:
    MYDIR = 'project_script'
    BASE_URL = 'http://hpc.sfpn.net/'
    PLUGINS = []
    COMMANDS = {}
    
    def ensure_mydir(self):
        if not os.path.exists(self.MYDIR):
            print("Creating the ``{}'' directory".format(self.MYDIR))
            os.mkdir(self.MYDIR)
    
    def web_request(self, url):
        request = urllib.request.Request(self.BASE_URL + url, method='GET')
        with urllib.request.urlopen(request, None, 10.0) as connexion:
            return connexion.read() 
    
    def get_manifest(self,):
        payload = self.web_request('manifest.json')
        return json.loads(payload)
    
    def get_file(self, url, target_filename):
        payload = self.web_request(url)
        with open(target_filename, 'wb') as f:
            f.write(payload)
    
    def refresh(self, m):
        change = False
        for filename, d in m.items():
            fetch = None
            if not os.path.exists(filename):
                fetch = "Getting {:32} ({}) [v{}]".format(filename, d['description'], d['version'])
            elif file_hash(filename) != d['hash']:
                fetch = "Upgrading {:32} ({}) [v{}]".format(filename, d['description'], d['version'])
            if fetch:
                if not change:
                    print("\n----------------------------------- UPGRADE ------------------------------------\n")
                print(fetch)
                self.get_file(d['url'], filename)
                change = True
        if change:
            print("\n--------------------------------------------------------------------------------\n")
    
    def import_module(self, name):
        return importlib.import_module(self.MYDIR + '.' + name)

    def load(self):
        files = glob.glob(self.MYDIR + '/cmd*.py')
        for filename in files:
            name, _ = os.path.splitext(os.path.relpath(filename, start=self.MYDIR))
            plugin = self.import_module(name)
            plugin.setup(self)
            self.PLUGINS.append(plugin)

    def usage(self):
        print("This script helps you complete the HPC home assignement.")
        print()
        print("Type 'project.py <subcommand> to trigger a specific action")
        print()
        print("Available commands:")
        print()
        for plugin in self.PLUGINS:
            plugin.usage()
        print()
        sys.exit()

    def run(self):
        self.ensure_mydir()
        m = self.get_manifest()
        self.refresh(m)
        self.load()
        if len(sys.argv) == 1:
            self.usage()
        args = sys.argv[1:]
        node = self.COMMANDS
    
        while True:
            if len(args) == 0:
                cmd = None
                args = []
            else:
                cmd = args[0]
                args = args[1:]
            if cmd not in node:
                print("No match for {} in {}".format(cmd, node))
                print("Unknown command ``{}''".format(' '.join(sys.argv[1:])))
                sys.exit()
            node = node[cmd]
            if callable(node):
                node(*args)
                break

if __name__ == '__main__':
    Main().run()