# -*- coding: utf-8 -*-


class PrintTXT:
    def __init__(self):
        self.list = []

    def add(self, message):
        message = message + '\n'
        self.list.append(message)

    def save(self, save_path):
        with open(save_path, 'w') as f:
            for txt in self.list:
                f.write(txt)
                print(txt)
