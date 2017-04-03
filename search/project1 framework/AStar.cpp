#include "Astar.h"
#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include <random>

using namespace std;

char AStarNode[6] = { 'A', 'B', 'C', 'D', 'E', 'F' };


void createGraph(Graph &G)
{
	cout << "enter the number of nodes in the graph:" << endl;
	cin >> G.n;
	cout << "enter the number of edges in the graph:" << endl;
	cin >> G.e;

	int i, j;
	int s, t; //start and end node number
	int v;  //value of edge between node s and t
	for (i = 0; i<G.n; i++)
	{
		G.H[i] = 0;
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

		cout << "next:" << endl;
	}
	cout << "enter heuristic val" << endl;
	for (int i = 0; i < G.n; i++)
	{
		cin >> G.H[i];
	}
}
bool checkInCloseList(GraphNode* curr, vector<GraphNode*> close_list)
{
	assert(curr != nullptr);
	///For close_list is vector:
	for (int i = 0; i<(int)close_list.size(); i++)
	{
		if (*curr == *(close_list[i]))
		{
			free(curr);

			return true;
		}
	}

	return false;
}
bool checkInOpenList(GraphNode* curr,
	priority_queue<GraphNode*, vector<GraphNode*>, cmpLarge>openlist,
	GraphNode*& found)
{
	assert(curr != nullptr);
	bool flag = false;
	vector<GraphNode*> backup;

	while (!openlist.empty())
	{
		backup.push_back(openlist.top());
		assert(openlist.top() != nullptr);
		//		cerr<< *(openlist.top()) << *curr << endl<<endl;
		if (*(openlist.top()) == *curr)
		{
			flag = true;
			found = openlist.top();
		}
		openlist.pop();
	}
	for (int i = 0; i<(int)backup.size(); ++i)
	{
		openlist.push(backup[i]);
	}
	return flag;
}
vector<GraphNode*> getSuccessors(const Graph& graph, GraphNode* curr)
{
	vector<GraphNode*> res;
	for (int i = 0; i < graph.n; i++)
	{
		if (graph.edges[curr->nodeNum][i] != 0)
		{
			GraphNode* tp = (GraphNode*)malloc(sizeof(GraphNode));
			res.push_back(tp);
			tp->nodeNum = i;
			tp->h = graph.H[i];
			tp->g = curr->g + graph.edges[curr->nodeNum][i];
			tp->preGraphNode = curr;
		}
	}
	return res;
}
void AStarShortestPathSearch(const Graph& graph, GraphNode& initNode, GraphNode& finalNode, vector<int> &resultPath){

	//push states to priority queue,and pop according to priority
	priority_queue<GraphNode*, vector<GraphNode*>, cmpLarge> openlist;
	vector<GraphNode*> closelist;
	openlist.push(&initNode);
	GraphNode* current;
	while (!openlist.empty())
	{
		current = openlist.top();
		openlist.pop();

		if (*current == finalNode)
		{
			GraphNode* tp=current;
			resultPath.push_back(tp->nodeNum);
			while (tp->nodeNum!=initNode.nodeNum)
			{
				tp = tp->preGraphNode;
				resultPath.push_back(tp->nodeNum);
			}
			reverse(resultPath.begin(), resultPath.end());
			return;
		}

		closelist.push_back(current);
		vector<GraphNode*> successors;
		successors = getSuccessors(graph,current);
		for (int i = 0; i < successors.size(); i++)
		{
			if (checkInCloseList(successors[i], closelist))
				continue;
			GraphNode* found;
			if (checkInOpenList(successors[i], openlist, found))
			{
				if (successors[i]->g <= found->g)
					openlist.push(successors[i]);
				else
					free(successors[i]);
			}
			else
			{
//				make_heap(const_cast<GraphNode**>(&openlist.top()), const_cast<GraphNode**>(&openlist.top()) + openlist.size(),
//					cmpLarge());
				openlist.push(successors[i]);
			}
				
		}
	}
	/*
	openlist.push(&initNode);
	while openlist is not empty:

	current = openlist.top()
	openlist.pop();

	if (current==goal node)
	return reconstruct_path(current, initNode, resultPath)

	closelist.push_back(current)

	successors=getSuccessors(current)
	for (i = 0; i < successors.size(); ++i)

	if successors[i] in closelist //if the successor has already been explored
	continue;
	if successors[i] not in openlist //check if the successor is not in the openlist
	compute g, h and preGraphNode of successors[i]
	push successors[i] to openlist

	else if g of successors[i] from current path is bigger than previous g of successors[i]    //remember to set proper value to g and h
	continue;

	update g, h and preGraphNode of successors[i]
	*/
}

bool checkGoalNode(const GraphNode* resultState, const GraphNode* goalNode){
	if (resultState->nodeNum == goalNode->nodeNum)
		return true;

	return false;
}

/* display the search result
resultPath: a vector which stores the node numbers in order along the shortest path found
MGraph : the search graph, here it is used to compute the path length
*/
bool displaySearchPath(const vector<int> &resultPath, const Graph& MGraph){
	int shortestPath = 0;
	int num = int(resultPath.size());
	if (resultPath.empty())
		return false;
	cout << "The shortest path is:" << endl;
	for (int i = 0; i < num - 1; ++i){
		cout << AStarNode[resultPath[i]] << "-> ";
		shortestPath += MGraph.edges[resultPath[i]][resultPath[i + 1]];

	}
	cout << AStarNode[resultPath[num - 1]] << endl;

	cout << "Path length: " << shortestPath << endl;

	return true;
}



