import pickle
import os

# Image folder should be categorized by subfolder in the style: "object_attribute"
# ie, images should contain folders for "apple_sliced, apple_whole, steak_cooked, etc."
# Each folder should contain the image files it categorizes.

imageFP = "rhys-lab\\images\\" # Filepath to image folder

folders = [x[0] for x in os.walk(imageFP)]
holder = []

for i in range(len(folders)):
    folders[i] = folders[i].replace(imageFP, "")
    if folders[i] != "":
        holder.append(folders[i])

folders = holder

print(folders)


data = {'annots':[], 'attributes':[], 'files':[], 'objects':[], 'pairs':[]}
print(data)

f = []
for i in folders:
    f.append(i)
    out = i.split("_")
    data['objects'].append(out[0])
    data['attributes'].append(out[1])

data['objects'] = list(set(data['objects']))
data['attributes'] = list(set(data['attributes']))

for i in range(len(data['attributes'])):
    for j in range(len(data['objects'])):
        p = data['objects'][j] + "_" + data['attributes'][i]
        if p in f:
            data['pairs'].append([i,j])


for path, subdirs, files in os.walk(imageFP):
    for name in files:
        st = os.path.join(path, name)
        st = st.replace(imageFP,"")
        oa = st.split("\\")[0]
        o, a = oa.split("_")

        attr = data['attributes'].index(a)
        obj = data['objects'].index(o)
        pair = data['pairs'].index([attr, obj])

        data['annots'].append({'pair': pair, 'obj': obj, 'attr': attr})
        data['files'].append(st)

pickle.dump( data, open( "metadata.pkl", "wb" ) )
