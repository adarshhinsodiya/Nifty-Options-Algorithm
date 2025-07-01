"""
Test script to verify Breeze API credentials.
Run with: python test_breeze_api.py
"""
import os
import sys
from dotenv import load_dotenv
from breeze_connect import BreezeConnect

def test_api_connection(api_key, api_secret, session_token, api_url):
    """Test connection to Breeze API with provided credentials."""
    print("\n=== Testing Breeze API Connection ===")
    print(f"API Key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")
    print(f"Session Token: {session_token}")
    
    try:
        # Initialize BreezeConnect
        breeze = BreezeConnect(api_key=api_key)
        breeze.root_url = api_url
        
        # Test 1: Try to get customer details with session token
        print("\nTest 1: Getting customer details with session token...")
        try:
            if session_token:
                # For older versions of breeze-connect, we set the session token directly
                breeze._session_token = session_token
                details = breeze.get_customer_details(session_token)
                print("Success! Customer details:")
                print(f"Name: {details.get('first_name', 'N/A')} {details.get('last_name', 'N/A')}")
                print(f"Email: {details.get('email_id', 'N/A')}")
                return True
            else:
                print("No session token provided, skipping session test")
        except Exception as e:
            print(f"Error with session token: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment
    api_key = os.getenv('BREEZE_API_KEY')
    api_secret = os.getenv('BREEZE_API_SECRET')
    session_token = os.getenv('BREEZE_SESSION_TOKEN')
    api_url = os.getenv('BREEZE_API_URL', 'https://api.icicidirect.com/breezeapi/api/v2/')
    
    # Validate we have minimum required credentials
    if not api_key:
        print("Error: BREEZE_API_KEY not found in .env file")
        return
    
    if not api_secret and not session_token:
        print("Error: Need either BREEZE_API_SECRET or BREEZE_SESSION_TOKEN in .env file")
        return
    
    # Run tests
    success = test_api_connection(api_key, api_secret, session_token, api_url)
    
    # Print summary
    print("\n=== Test Summary ===")
    if success:
        print("Tests completed successfully!")
    
    print("\nNote: If you're getting authentication errors:")
    print("1. Verify your API key and secret in the ICICI Direct portal")
    print("2. Make sure your account has API access enabled")
    print("3. Check if your IP is whitelisted in the ICICI Direct portal")
    print("4. Try generating a new session token from the ICICI Direct website")

if __name__ == "__main__":
    main()
