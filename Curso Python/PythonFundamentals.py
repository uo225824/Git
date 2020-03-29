#Definir funciones

def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1,2))
print(add_numbers(1,2,3))

def add_numebers1(x,y,z=None,Flag=False):
    if  (Flag==True):
        print("Flag is true")
    if  (z==None):
        x+y
    else:
        x+y+z

print(add_numebers1(1,2,3,True))

a=add_numbers

print(a(1,2))


#Types and Sequences

type('This is a string')
type(1.0)
print(type(1.))

#listas

x=[1,'a',2,'b',3]
print(x)
x.append(3.3)
print(x)


for i in x:
    print(i)

i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1


[1,2] + [3,4]
[1]*3
1 in [1, 2, 3]

x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters

x[-4:-2]
x[3:]


firstname = 'Christopher'
lastname = 'Brooks'

print(firstname + ' ' + lastname)
print(firstname*3)
print('Chris' in firstname)


firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)


#Diccionarios

print('DICCIONARIOS')
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator

for name in x:
    print(x[name])

for email in x.values():
    print(email)

for name, email in x.items():
    print(name)
    print(email)

x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x

print(fname)

#Strings

print('Chris'+str(2))

sales_record={
    'price':3.24,
    'num_item':4,
    'person':'Chris'
}

sales_statement='{} bought {} items at price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_item'],
                             sales_record['price'],
                             sales_record['num_item']*sales_record['price']))


#Fechas y tiempo

import datetime as dt
import time as tm

print(tm.time())

dtnow=dt.datetime.fromtimestamp(tm.time())
print(dtnow)

delta = dt.timedelta(days = 100) # create a timedelta of 100 days
print(delta)


#Objetos y mapas

class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location

person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
print(cheapest)

for item in cheapest:
    print(item)


#Funciones lambda y compresion de listas


my_function = lambda a, b, c : a + b
my_function(1, 2, 3)


my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
my_list


my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list


#Numpy


