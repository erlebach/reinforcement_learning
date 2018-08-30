using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

//class to store past states for actions and associated rewards
public class Replay
{
    public List<double> states;
    public double reward;

	public Replay(double xr, double ballz, double ballvz, double r)
	{
		states = new List<double>();
		states.Add(xr);
		states.Add(ballz);
		states.Add(ballvz);
		reward = r;
	}
}

public class Brain : MonoBehaviour {
	public GameObject ball;

	ANN ann;

	float reward = 0.0f;							//reward to associate with actions
	List<Replay> replayMemory = new List<Replay>();	//memory - list of past actions and rewards
	int mCapacity = 10000;							//memory capacity
	
	// 1 hidden layer with six neurons and discount=1.0 ==> solve problem in < 20 fails
	float discount = 1.0f; //0.99f;							//how much future states affect rewards [.99f]
	float exploreRate = 100.0f;						//chance of picking random action
	float maxExploreRate = 100.0f;					//max chance value
    float minExploreRate = 0.01f;					//min chance value
    float exploreDecay = 0.0001f;					//chance decay amount for each update [0.0001f]

	Vector3 ballStartPos;
	int failCount = 0;								//count when the ball is dropped
	float tiltSpeed = 0.5f;							//max angle to apply to tilting each update
													//make sure this is large enough so that the q value
													//multiplied by it is enough to recover balance
													//when the ball gets a good speed up
	float timer = 0;								//timer to keep track of balancing
	float maxBalanceTime = 0;						//record time ball is kept balanced	


	// with 10 neurons in hidden layer, learns faster, but less stable
	// with 15 neurons, hard to find a balance (> 40 iter)
	// with 2 layers of 3 neurons convergence is slower. Why? 
	// 2 layers of 2 neurons: convergence even slower, but trials are much more dyanmic. 
	//   How to visualize these different effects? 
	// How to do RL together with regularization
	void Start () {
		// ANN(int nI, int nO, int nH, int nPH, double a)
		ann = new ANN(3,2,1,6,0.2f); // orig: 3,2,1,6,0.2f
		ballStartPos = ball.transform.position;
		Time.timeScale = 5.0f;
	}

	GUIStyle guiStyle = new GUIStyle();
	void OnGUI()
	{
		guiStyle.fontSize = 25;
		guiStyle.normal.textColor = Color.white;
		GUI.BeginGroup (new Rect (10, 10, 600, 150));
		GUI.Box (new Rect (0,0,140,140), "Stats", guiStyle);
		GUI.Label(new Rect (10,25,500,30), "Fails: " + failCount, guiStyle);
		GUI.Label(new Rect (10,50,500,30), "Decay Rate: " + exploreRate, guiStyle);
		GUI.Label(new Rect (10,75,500,30), "Last Best Balance: " + maxBalanceTime, guiStyle);
		GUI.Label(new Rect (10,100,500,30), "This Balance: " + timer, guiStyle);
		GUI.EndGroup ();
	}

	void Update()
	{
		if(Input.GetKeyDown("space"))
			ResetBall();
	}

	void FixedUpdate () {
		timer += Time.deltaTime;
		List<double> states = new List<double>();
		List<double> qs = new List<double>();
			
		states.Add(this.transform.rotation.x);
		states.Add(this.transform.position.z);
		states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x); // reflection not working in VCS
		
		qs = SoftMax(ann.CalcOutput(states));  // why a softmax?
		double maxQ = qs.Max();  // cost: O(L), where L is length of the list
		int maxQIndex = qs.ToList().IndexOf(maxQ);  // cost is O(L)
		// in my opinion, exploreRate should decrease after each fail and not after each fixedUpdate
		exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);

		// Udemy: remove these lines will accelerate convergence
		// more exporation early on, and less later on
		if(Random.Range(0,100) < exploreRate)
			maxQIndex = Random.Range(0,2);   // choose either 0 or 1

		if(maxQIndex == 0)
		    // public void Rotate(Vector3 eulerAngles, Space relativeTo = Space.Self);
			// public void Rotate(Vector3 axis, float angle, Space relativeTo = Space.Self);
			this.transform.Rotate(Vector3.right, tiltSpeed * (float)qs[maxQIndex]); 
		else if (maxQIndex == 1)
			this.transform.Rotate(Vector3.right, -tiltSpeed * (float)qs[maxQIndex]);
		
		if (ball.GetComponent<BallState>().dropped)
			reward = -1.0f;
		else
			reward = 0.1f;   // [0.1f]


		Replay lastMemory = new Replay(this.transform.rotation.x,
										ball.transform.position.z,
										ball.GetComponent<Rigidbody>().angularVelocity.x, 
										reward);

		if(replayMemory.Count > mCapacity)
			replayMemory.RemoveAt(0);
		
		replayMemory.Add(lastMemory);

		if(ball.GetComponent<BallState>().dropped) 
		{
			for(int i = replayMemory.Count - 1; i >= 0; i--)
			{
				List<double> toutputsOld = new List<double>();
				List<double> toutputsNew = new List<double>();
				toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));	// why a softmax?

				double maxQOld = toutputsOld.Max();
				int action = toutputsOld.ToList().IndexOf(maxQOld);

			    double feedback;
				if(i == replayMemory.Count-1 || replayMemory[i].reward == -1)
					feedback = replayMemory[i].reward;
				else
				{
					toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i+1].states));
					maxQ = toutputsNew.Max();
					feedback = (replayMemory[i].reward + 
						discount * maxQ);
				} 

				toutputsOld[action] = feedback;
				ann.Train(replayMemory[i].states,toutputsOld);
			}
		
			if(timer > maxBalanceTime)
			{
			 	maxBalanceTime = timer;
			} 

			timer = 0;

			ball.GetComponent<BallState>().dropped = false;
			this.transform.rotation = Quaternion.identity;
			ResetBall();
			replayMemory.Clear();
			failCount++;
		}	
	}

	void ResetBall()
	{
		ball.transform.position = ballStartPos;
		ball.GetComponent<Rigidbody>().velocity = new Vector3(0,0,0); // GE
		ball.GetComponent<Rigidbody>().angularVelocity = new Vector3(0,0,0);
	}

	List<double> SoftMax(List<double> oSums) 
    {
      double max = oSums.Max();
	  //print("softmax in: " + oSums.Count);

      float scale = 0.0f;
      for (int i = 0; i < oSums.Count; ++i)
        scale += Mathf.Exp((float)(oSums[i] - max));

      List<double> result = new List<double>();
      for (int i = 0; i < oSums.Count; ++i)
        result.Add(Mathf.Exp((float)(oSums[i] - max)) / scale);


	  //print("softmax out: " + result.Count);
      return result; 
    }
}
