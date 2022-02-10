using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Player : MonoBehaviour
{
    private bool jumpKey;
    private float horizontalInput;
    private Rigidbody rigidbodyComponent;
    public Transform groundCheck;
    public float speed;
    // Start is called before the first frame update
    void Start()
    {
        rigidbodyComponent = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        bool movingLeft = Input.GetKey(KeyCode.LeftArrow);
        bool movingRight = Input.GetKey(KeyCode.RightArrow);
        //check space key
        if (Input.GetKeyDown(KeyCode.Space))
        {
            jumpKey = true;
        }
        if (movingLeft){
            this.gameObject.transform.position += new Vector3(-1, 0, 0) * speed;
        }
        else if (movingRight){
            this.gameObject.transform.position += new Vector3(1, 0, 0) * speed;
        }
        horizontalInput = Input.GetAxis("Horizontal");
    }

    private void FixedUpdate()
    {
        // if (!isGrounded)
        // {
        //     return;
        // }
        // rigidbodyComponent.velocity = new Vector3(horizontalInput, GetComponent<Rigidbody>().velocity.y, 0);
        
        // if (movingLeft){
        //     this.gameObject.transform.position += new Vector3(-1, 0, 0) * speed;
        // }
        // else if (movingRight){
        //     this.gameObject.transform.position += new Vector3(1, 0, 0) * speed;
        // }
        //below is 1 because it always collides with the player
        //if a layer mask is used it will be dif ~ 1:40
        // if (Physics.OverlapSphere(groundCheck.position, 0.1f).Length == 1)
        // {
        //     return;
        // }

        // if (jumpKey)
        // {
        //     rigidbodyComponent.AddForce.parent(Vector3.up * 5, ForceMode.VelocityChange);
        //     jumpKey = false;
        // }
        
    }

    // void OnCollisionEnter(Collision collision)
    // {
    //     isGrounded = true;
    // }

    // void OnCollisionExit(Collision collision)
    // {
    //     isGrounded = false;
    // }
}
