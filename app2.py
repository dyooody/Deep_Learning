from flask import Flask, render_template, jsonify
import json
from os import listdir
from os.path import isfile, join
import random
from collections import defaultdict
import csv
from scipy import cluster # pip install scipy
import numpy as np # pip install numpy
import pandas as pd # pip install pandas

app = Flask(__name__)

@app.route("/clusters/<int:result_id>/<class_name>")
def get_clustering_results(result_id, class_name):
	return render_template('clusters.html', result_id=result_id, class_name=class_name)

@app.route("/hclusters/<int:result_id>/<class_name>")
def get_hierarchical_clustering_results(result_id, class_name):
	total_number_of_patches = 0
	with open("clustering_results/"+class_name+"_patch_info.csv", "r") as patch_file:
		patch_reader = csv.reader(patch_file, delimiter=',')
		for i, row in enumerate(patch_reader):
			if i > 0:
				total_number_of_patches += 1
	return render_template('hclusters.html', result_id=result_id, class_name=class_name, total_number_of_patches=total_number_of_patches)

@app.route("/data/pretest")
def temp_for_preprocessing():
	labels_dict = []
	for k in range(10):
		patch_path = "static/patches/reduced_dim5_numClusters10_cluster" + str(k)
		patch_files = [file_name for file_name in listdir(patch_path)]
		for file_name in patch_files:
			image_id = int(file_name[5:file_name.find(".")])
			labels_dict.append({"id": image_id, "cluster_no": k})
	with open("clustering_results/clusters_1_schoolbus.json", "w") as f:
		json.dump({"data": labels_dict}, f, indent=4)
	return json.dumps({"data": labels_dict})

@app.route("/data/<int:result_id>/<class_name>")
def get_data_for_clustering_results(result_id, class_name):	
	patch_path = "static/patches/" + class_name
	patch_files = [file_name for file_name in listdir(patch_path)]
	d = {"data": [{
			"id": int(file_name[5:file_name.find(".")]), 
			"filename": file_name, 
			"cluster_no": 0 # this needs to be extracted from csv in clustering_reuslts/
		} for file_name in patch_files]}
	#with open("clustering_results/results_"+str(result_id)+"_"+str(class_name)+".json", "r") as f:
	#	d = json.load(f)

	with open("clustering_results/clusters_1_"+class_name+".json", "r") as f:
		d = json.load(f)
		clusters = defaultdict(list)
		for x in d["data"]:
			clusters[x["cluster_no"]].append({
					"id": x["id"], 
					"filename": "/static/patches/"+class_name+"/patch"+str(x["id"])+".png"
				})
		return json.dumps({
				"clusters": [{
					"cluster_no": c_no, 
					"patches": clusters[c_no][:min(len(clusters[c_no]),50)]
				} for c_no in clusters]
			})

@app.route("/data_hclusters/<int:result_id>/<class_name>/<int:num_cluster>")
def get_data_for_hier_clustering_results(result_id, class_name, num_cluster):
	hclusters_filename = "clustering_results/hclusters_1_"+class_name+".csv"
	with open(hclusters_filename, "r") as file:
		reader = csv.reader(file, delimiter=',')
		
		linkage = []
		for i, row in enumerate(reader):
			if i>0:
				linkage.append([int(row[1]), int(row[2]), float(row[3]), int(row[4])])
		# print(linkage)

		Z = np.array(linkage)

		# optimal order exists. want to know an index given a patch id
		leaf_order_index = dict([(l, i) for i, l in enumerate(cluster.hierarchy.leaves_list(Z))])
		
		cluster_assignments = [{"patch_index": i, "cluster_no": x[0], "left_order_no": leaf_order_index[i]} 
			for i, x in enumerate(cluster.hierarchy.cut_tree(Z, n_clusters=[num_cluster]))]
		# print(cluster_assignments)
		# assigns a cluster number to each patch

		# sort based on left order
		cluster_assignments = sorted(cluster_assignments, key=lambda x: x["left_order_no"])

		# load data from patch index csv file
		patches = []
		patch_info_filename = "clustering_results/"+class_name+"_patch_info.csv"
		with open(patch_info_filename, "r") as patch_file:
			patch_reader = csv.reader(patch_file, delimiter=',')
			for i, row in enumerate(patch_reader):
				if i > 0:
					# need only patch index and updated image path
					patches.append([int(row[2]), str(row[0])])

		# merge the loaded file with the "cluster_assignments" so that
		# for each patch index, we know where the file is and what its cluster no is
		image_patches_with_cluster_assignment_info = []
		for item in cluster_assignments:
			# each row in patches has a unique patch index I think
			# just match patch filename to cluster number using patch_index as link?

			# assumes patch index is 0-4174, like in csv, and that patch index
			# matches up with index of patches array
			# get only the unique part of the filepath + '.png'
			filename = patches[item["patch_index"]][1].split('/')[4]
			image_patches_with_cluster_assignment_info.append(
				{"cluster_no": item["cluster_no"], 
				 "patch_index": item["patch_index"],
				 "patch_filename": filename})

		# image_patches_with_cluster_assignment_info = [...] (size of array = num of rows for the csv file)


		# you will likely need to create the following format as an intermediate step
		clusters = defaultdict(list)
		for x in image_patches_with_cluster_assignment_info:
			clusters[int(x["cluster_no"])].append({
					"id": int(x["patch_index"]), 
					"filename": "/static/patches/"+class_name+"/patch_images/"+str(x["patch_filename"])
				})

		# output format would be like:		
		output_data = {
				"total_number_of_patches": len(cluster_assignments),
				"clusters": [{
					"cluster_no": c_no, 
					"patches": clusters[c_no][:min(len(clusters[c_no]),30)]
				} for c_no in clusters]
			}

		return json.dumps(output_data)


@app.route("/sag/<int:image_id>")
def get_sag_tree_for_image(image_id):
	image_url = get_image_url(image_id)
	regions = get_sag_regions_for_image(image_id)
	return render_template('sag_for_image.html', image_id=image_id, image_url=image_url, regions=regions)

def get_image_url(image_id):
	lookup = pd.read_csv("datafiles/new_ids_all_src_images.csv")
	image_url = list(lookup.loc[lookup['source_image_id'] == image_id]['IMAGE PATH'])
	image_url = image_url[0].replace('./', '/static/')
	return image_url


@app.route("/data/sag-graph/<int:image_id>")
def get_sag_regions_for_image(image_id):
	df = pd.read_csv("datafiles/combined_region_images.csv")
	df = df[df['source_image_id']==image_id]
	print(df.shape)
	df = df.rename(columns={"CONFIDENCE": "confidence", "ACTIVE GRID CELLS": "active_cells", "REGION IMAGE PATH": "region_path", "IS_ROOT": "is_root", 'NUMBER OF ACTIVE CELLS': "num_of_cells", 'CLASS' : "class"})
	df = df[["region_id", "source_image_id", "confidence", "active_cells", "region_path", "is_root", "num_of_cells"]]

	df = df.sort_values(by=["is_root", "confidence"], ascending=[False, False])
	df = df.sort_values(by=['num_of_cells'], ascending=[False])
	df = df.drop_duplicates(subset = ['source_image_id', 'is_root', 'confidence', 
		'active_cells', 'num_of_cells', 'region_path'])
	print(df.shape)
	df["active_cells"].str[1:-1].tolist()
	df["region_path"] = df["region_path"].str[27:]
	regions = json.loads(df.to_json(orient="records"))

	#create Nodes
	nodes = []
	cnt = 0
	for nd in range(len(regions)):
		if float(regions[nd]['confidence']) > 0.4:
			node = {}
			node['region_id'] = regions[nd]['region_id']
			node['confidence'] = regions[nd]['confidence']
			node['active_cells'] = regions[nd]['active_cells']
			node['region_path'] = regions[nd]['region_path']
			node['is_root'] = regions[nd]['is_root']
			node['node_index'] = cnt
			cnt += 1

			nodes.append(node)	

	#create Links
	links = []
	for li in range(len(nodes)):
		grids = json.loads(nodes[li]['active_cells'])
		for nk in range(li+1, len(nodes)):
			grid = json.loads(nodes[nk]['active_cells'])
			res = all(elem in grids for elem in grid)
			if res == True and (len(grids) - len(grid) == 1) :
				link = {}
				link['source'] = nodes[li]['node_index']
				link['target'] = nodes[nk]['node_index']
				links.append(link)
	

	# NEED TO BE EDITED HERE
	regions_in_tree = {"nodes": nodes, "links": links}
	
	#return json.dumps({
	#	"regions": regions,
	#	"regions_in_tree": regions_in_tree
	#})
	return jsonify(regions_in_tree)


@app.route("/sag/<class_name>")
def get_sag_page_for_class(class_name):
	return render_template('sag_class_overview.html', class_name=class_name)

@app.route("/sag/<class_name>/data")
def get_sag_images(class_name):
	df = pd.read_csv("static/"+class_name+"/sag_regeneration.csv")
	df = df.drop_duplicates()
	df = df.sort_values(by="CONFIDENCE")

	top_ten = list(df["REGION IMAGE PATH"].head(50)) 

	return json.dumps({
		"images": [x.replace("./sag_regeneration_dataset", "/static") for x in top_ten]
	})


if __name__ == "__main__":
	app.run()
