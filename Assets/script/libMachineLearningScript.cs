using UnityEngine;

public class libMachineLearningScript: MonoBehaviour
{
    void Start()
    {
        Debug.Log("With VS DLL : ");
        Debug.Log(libMachineLearningWrapper.my_add(42, 51));
        Debug.Log(libMachineLearningWrapper.my_mul(2, 3));
    }
    // Update is called once per frame
    void Update()
    {
    }
}