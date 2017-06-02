import urllib, json, cPickle

f = open('city_name_pinyin.txt')
city_name = f.readlines()
city_name = [name.strip(' \t\r\n') for name in city_name]
f.close()
print city_name

key = 'AIzaSyDw12ZWkEypDDOJMbT3UPVouVQLLppeimM'
# address='Beijing'
location = []
for address in city_name:
    my_url = (
                 'https://maps.googleapis.com/maps/api/geocode/json?address=%s&key=%s'
             ) % (address, key)
    print my_url
    try:
        response = urllib.urlopen(my_url)
    except urllib.error.HTTPError:
        print "Check Internet"
    else:
        print "Unkown"
    json_data = json.loads(response.read())
    print json_data

    loc = [json_data['results'][0]['formatted_address'],
           json_data['results'][0]['geometry']['location']]
    print loc
    location.append(loc)

print len(location), location
with open("location.pkl", 'w') as f:
    cPickle.dump(location, f)
