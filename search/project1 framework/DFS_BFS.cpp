#include "DFS_BFS.h"
#include <iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<queue>
#include<stack>
#include <unordered_set>
#include "Astar.h"

using namespace std;

bool vis[maxn];  //for each node in the graph, mark whether the node has been visited during searching.
char node[6] = { 'A', 'B', 'C', 'D', 'E', 'F' };  //node symbol in order, used for visualization of searching result.


void createMGraph(MGraph &G) 
{
	cout << "enter the number of nodes in the graph:" << endl;
	cin >> G.n;
	cout << "your input is " << G.n << endl;
	cout << "enter the number of edges in the graph:" << endl;
	
	cin >> G.e;
	cout << "your input is " << G.e << endl;
	int i, j;
	int s, t;  //start and end node number
	int v;  //value of edge between node s and t
	for (i = 0; i<G.n; i++)  
	{
		vis[i] = false;
		for (j = 0; j<G.n; j++)
		{
			G.edges[i][j] = 0;
		}
	}
	cout << "enter the value of edge/weights and corresponding node number" << endl;
	cout << "(e.g. 1 3 10 means the edge from node 1 to node 3 is 10) : " << endl;
	cout << "first: " << endl;
	for (i = 0; i<G.e; i++) //set value for non-zero edges
	{
		cin >> s >> t >> v;    
		G.edges[s][t] = v;
		cout <<"your input is "<<" "<< s<<" " << t<<" " << v<<endl;
		cout << "next:" ;
	}
}
/* Your implmentation of DFS, BFS for searching 
*/

/* Recursive implementation of DFS for finding path from given start node to end node
G: the graph for searching
start: start node number for searching
end: end node number for searching
endflag: a bool variable for determining whether searching has reached the end node
*/
void dfs(MGraph G, int start, int end, bool& return_flag,vector<int>& path)
{
	if (return_flag == true)
		return;
	
	if (start == end)
	{
		cout << "!!DFS find Route: ";
		for (int i = 0; i < (int)path.size(); ++i)
		{
			cout << node[ path[i]] << " ";
		}
		cout<< endl;
		return_flag = true;
		return;
	}
	else
	{
		vis[start] = true;
		for (int i = 0; i < G.n; i++)
		{
			if (G.edges[start][i] != 0 && vis[i]==false)
			{
				path.push_back(i);
				dfs(G, i, end, return_flag,path);
				path.pop_back();
			}
		}
	}
	/* if endflag is true: return;

	print (start node);
	set vis[start] to true;

	for (the neiboring nodes of start node):
	  if not visited:
	      dfs(G, neighbor,end,endflag);
		  if neighbor=end node:
		     set endflag true;
			 return;
	*/
}

/* Non-recursive implementation of DFS for finding path from given start point to end point
   data structure: Stack (LIFO)
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void dfs1(MGraph G, int start, int end)
{
	stack<int> s;
	bool vis_dfs1[maxn] = { false };
	vis_dfs1[start] = true;
	int index = -1;
	vector<int> path;

	if (start == end) return;
	s.push(start);
	path.push_back(start);
	while (!s.empty()){
		for (int i = 0; i < G.n; i++){
			int top = s.top();
			if (G.edges[top][i] == 0) continue;
			if (i == end)
			{
				path.push_back(i);
				cout << "!!DFS1 find Route(use stack): ";
				for (int i = 0; i < (int)path.size(); ++i)
				{
					cout << node[path[i]] << " ";
				}
				cout << endl;
				return;
			}
			if (!vis_dfs1[i] && i != index)
			{
				path.push_back(i);
				vis_dfs1[i] = true;
				s.push(i);
				continue;
			}
			if (i == G.n) index = top;
		}
		s.pop();
		path.pop_back();
	}
	return;
	/*
	check if start node is end node;

	s.push(start);  //push start node into stack

	while (!s.empty())

	get top node in the stack;

	for (all the neighboring nodes) //访问与顶点i相邻的顶点

	if (not visited)

	operations to the node
	*/
}

/* BFS implementation for finding path from given start point to end point
   data structure: Queue (FIFO)
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void bfs(MGraph G, int start, int end)
{
	queue<vector<int>> Q;

	/*
	set vis[start] to true;
	Q.push(start);  push start node to stack

	while (!Q.empty())
	
		get top node;

		for (all the neighbor node of top node) 
		  //BFS traversal
	
	*/
	bool vis_bfs[maxn]={false};
	vis_bfs[start] = true;
	vector<int> now_path;
	now_path.push_back(start);
	Q.push(now_path);
	
	while (!Q.empty())
	{
		vector<int> now_path = Q.front();
		Q.pop();
		if (now_path.back() == end){
			cout << "!!Solution find by BFS: ";
			for (auto it = now_path.begin(); it+1 != now_path.end(); ++it)
				cout << node[*it] << " --> ";
			cout << node[ now_path.back()];
			cout << endl;
			return;
		}
		for (int j = 0; j < G.n; j++)
		{
			if (G.edges[now_path.back()][j] != 0 && vis_bfs[j]==false){
				vector<int> new_path(now_path);
				new_path.push_back(j);
				Q.push(new_path);
				vis_bfs[j] = true;
			}

		}
	}
}

