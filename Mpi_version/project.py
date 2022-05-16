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
        action = False
        restrained = False
        for filename, d in m.items():
            msg = None
            if not os.path.exists(filename):
                msg = "Getting {:32} ({}) [v{}]".format(filename, d['description'], d['version'])
                download = True
            elif file_hash(filename) != d['hash']:
                download = 'autoupdate' in d
                if download:
                    msg = "Upgrading {:32} ({}) [v{}]".format(filename, d['description'], d['version'])
                else:
                    msg = "***NOT*** upgrading {:32} ({}) [v{}]".format(filename, d['description'], d['version'])
            if msg:
                if not action:
                    print("\n----------------------------------- UPGRADE ------------------------------------\n")
                print(msg)
                if download:
                    self.get_file(d['url'], filename)
                else:
                    restrained = True
                action = True
        if restrained:
            print(f"\n!!! WARNING !!! Some of your files differ from those on the server.")
            print("                 Either they have been remotely updated or you changed them locally.") 
            print("                 Your files have not been modified.") 
            print("                 Rename or delete the old file to get the version from the server.")

        if action:
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