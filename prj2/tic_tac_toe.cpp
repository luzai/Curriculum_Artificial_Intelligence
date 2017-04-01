#include "tic_tac_toe.hpp"
#include "iostream"
#include "string"
#include <vector>
using namespace std;

/*	Search the next step by MinMaxSearch with depth limited strategy.
	currentState is the current state of the game, see structure TicTacToeState for more details.
		WARNING: The search depth is limited to 3, computer uses circles(1).
	r and c are returned row and column index that computer player will draw a circle on. 0<=r<=2, 0<=c<=2
	Your code starts here
*/
void miniMaxSearchForTicTacToe(const TicTacToeState& currentState, int& r, int& c)
{
	vector<Action> actions;
	vector<int> values;
	getAvailableActions(currentState, actions);
	for (int i = 0; i < (int)actions.size(); ++i)
	{
		values.push_back(MIN_Value(actionResult(currentState, actions[i], 1), currentState));
	}
	int max_idx = 0;
	int max_val = values[max_idx];
	for (int i = 1; i < (int)values.size(); ++i)
	{
		if (values[i] > max_val)
		{
			max_idx = i;
			max_val = values[i];
		}
	}
	r = actions[max_idx].row;
	c = actions[max_idx].col;
}



// You do not need the following code and do not revise it.
GameJudge::GameJudge()
{
	this->gameState = TicTacToeState();
}

void GameJudge::makeAMove(const int& r, const int& c, const int player)
{
	//player = 1 for computer, play = -1 for human
	//1 stands for circle, -1 stands for cross, 0 stands for empty space
	// computer uses circle, human uses cross
	this->gameState.state[r][c] = player;
}

/*	Return game status
	1 for computer wins
	-1 for human wins
	0 for draw
	2 in the play
*/
int GameJudge::checkGameStatus()
{

	// All spaces are occupied but nobody wins
	int numSpaces = 0;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; j++)
		{
			if (this->gameState.state[i][j] == 0)
			{
				numSpaces ++;
			}
		}
	}
	if (numSpaces == 0)
	{
		return 0;
	}

	// somebody wins
	for (int r = 0; r < 3; r++)
	{
		int sum_row = this->gameState.state[r][0] +
			this->gameState.state[r][1]+ this->gameState.state[r][2];
		if (sum_row == 3)
		{
			return 1;
		}
		if (sum_row == -3)
		{
			return -1;
		}
	}
	for (int c = 0; c < 3; c++)
	{
		int sum_col = this->gameState.state[0][c] +
			this->gameState.state[1][c]+ this->gameState.state[2][c];
		if (sum_col == 3)
		{
			return 1;
		}
		if (sum_col == -3)
		{
			return -1;
		}
	}
	int sum_diag = this->gameState.state[0][0] +
		this->gameState.state[1][1]+this->gameState.state[2][2];
	if (sum_diag == 3)
	{
		return 1;
	}
	if (sum_diag == -3)
	{
		return -1;
	}
	sum_diag = this->gameState.state[0][2] +
		this->gameState.state[1][1]+this->gameState.state[2][0];
	if (sum_diag == 3)
	{
		return 1;
	}
	if (sum_diag == -3)
	{
		return -1;
	}

	// in the play
	return 2;
}

void GameJudge::humanInput(int& r, int& c)
{
	cout << "Input the row and column index of your move " << endl;
	cout << "1,0 means draw a cross on the row 1, col 0" << endl;
	string str;
	bool succ = false;
	while(!succ)
	{
		cin >> str;
		sscanf(str.c_str(), "%d,%d", &r, &c);
		if (r < 0 || r > 2 || c < 0 || c > 2)
		{
			succ = false;
			cout << " Invalidate input, the two numbers should >> 0 and << 2" << endl;
		}
		else if (this->gameState.state[r][c] != 0)
		{
			succ = false;
			cout << " You can not put cross on this place " << endl;
		}
		else
		{
			succ = true;
		}
	}
}

void GameJudge::printStatus(const int player, const int status)
{
	cout << "------------------------------" << endl;
	for (int r= 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			if (this->gameState.state[r][c] == 1)
			{
				cout << "[O]";
			}
			if (this->gameState.state[r][c] == -1)
			{
				cout << "[X]";
			}
			else if(this->gameState.state[r][c] == 0)
			{
				cout << "[ ]";
			}

		}
		cout << endl;
	}
	if (player == 1)
	{
		cout << "Last move was conducted by computer " << endl;
	}
	else if (player == -1)
	{
		cout << "Last move was conducted by you " << endl;
	}
	if (status == 1)
	{
		cout << "Computer wins " << endl;
	}
	else if (status == -1)
	{
		cout << " You win " << endl;
	}
	else if (status == 2)
	{
		cout << " Game going on " << endl;
	}
	else if (status == 0)
	{
		cout << " Draw " << endl;
	}
}
TicTacToeState GameJudge::getGameStatus()
{
	return this->gameState;
}