class HelloWorld(object):

    __age = 0

    def __init__(self, age):
        self.__age = age
        print("Hello Wooooorld")

    def getAge(self):
        return self.__age
    