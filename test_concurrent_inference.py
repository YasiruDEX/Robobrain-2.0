#!/usr/bin/env python3
"""
Test script to verify concurrent inference handling.
This simulates multiple frontend requests hitting the backend simultaneously.
"""

import requests
import time
import concurrent.futures
from typing import List, Dict
import sys

BASE_URL = "http://localhost:5001"

def check_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Backend is healthy")
            print(f"  Model loaded: {data['model_loaded']}")
            print(f"  Active sessions: {data['active_sessions']}")
            return data['model_loaded']
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to backend: {e}")
        print(f"  Make sure backend is running: python backend.py")
        return False

def create_session() -> str:
    """Create a new chat session."""
    response = requests.post(f"{BASE_URL}/api/session")
    data = response.json()
    session_id = data.get('session_id') or data.get('sessionId')
    print(f"✓ Created session: {session_id}")
    return session_id

def send_chat_request(session_id: str, message: str, request_num: int) -> Dict:
    """Send a chat request and measure response time."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                'session_id': session_id,
                'message': message,
                'task': 'general',
                'enable_thinking': True
            },
            timeout=60  # 60 second timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', 'No answer')
            return {
                'success': True,
                'request_num': request_num,
                'elapsed': elapsed,
                'answer_length': len(answer),
                'has_error': 'error' in data
            }
        else:
            return {
                'success': False,
                'request_num': request_num,
                'elapsed': elapsed,
                'error': f"Status {response.status_code}: {response.text}"
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'request_num': request_num,
            'elapsed': elapsed,
            'error': str(e)
        }

def test_sequential_requests(session_id: str, num_requests: int = 3):
    """Test sequential requests (baseline)."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Sequential Requests (Baseline)")
    print(f"{'='*60}")
    
    results = []
    total_start = time.time()
    
    for i in range(num_requests):
        print(f"\n[Request {i+1}/{num_requests}] Sending...")
        result = send_chat_request(
            session_id, 
            f"Describe this scene briefly. Request {i+1}.",
            i+1
        )
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Completed in {result['elapsed']:.2f}s")
            print(f"  Answer length: {result['answer_length']} chars")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"Sequential Test Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg per request: {total_time/num_requests:.2f}s")
    successful = sum(1 for r in results if r['success'])
    print(f"  Success rate: {successful}/{num_requests}")
    print(f"{'='*60}")
    
    return results

def test_concurrent_requests(session_id: str, num_requests: int = 5):
    """Test concurrent requests (should be serialized by backend)."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Concurrent Requests (Serialization Test)")
    print(f"{'='*60}")
    print(f"Launching {num_requests} concurrent requests...")
    print(f"Backend should serialize them to prevent GPU OOM\n")
    
    total_start = time.time()
    
    # Launch all requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(
                send_chat_request,
                session_id,
                f"What can you tell me about this image? Request {i+1}.",
                i+1
            )
            futures.append(future)
        
        # Wait for all to complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['success']:
                print(f"  ✓ Request {result['request_num']} completed in {result['elapsed']:.2f}s")
            else:
                print(f"  ✗ Request {result['request_num']} failed: {result['error']}")
    
    total_time = time.time() - total_start
    
    # Sort results by request number for better readability
    results.sort(key=lambda x: x['request_num'])
    
    print(f"\n{'='*60}")
    print(f"Concurrent Test Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg per request: {total_time/num_requests:.2f}s")
    successful = sum(1 for r in results if r['success'])
    print(f"  Success rate: {successful}/{num_requests}")
    
    if successful == num_requests:
        print(f"\n  🎉 SUCCESS! All concurrent requests handled without OOM!")
        print(f"  The threading lock is working correctly.")
    else:
        print(f"\n  ⚠ WARNING: {num_requests - successful} requests failed")
        print(f"  Check backend logs for details")
    
    print(f"{'='*60}")
    
    return results

def main():
    print("RoboBrain 2.0 - Concurrent Inference Test")
    print("="*60)
    
    # Check if backend is running
    print("\nStep 1: Checking backend health...")
    if not check_backend_health():
        print("\n❌ Backend is not ready. Please start it first:")
        print("   conda activate robobrain2-env")
        print("   python backend.py")
        sys.exit(1)
    
    # Create a test session
    print("\nStep 2: Creating test session...")
    try:
        session_id = create_session()
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        sys.exit(1)
    
    # Note: For these tests to work, you need to set an image first
    # You can do this by uploading an image and then referencing it
    # For now, we'll assume the backend can handle requests without explicit images
    # (though actual inference might fail - we're testing OOM prevention, not functionality)
    
    print("\nℹ Note: These tests will fail inference if no image is set,")
    print("  but they will test that concurrent requests don't cause GPU OOM.")
    print("  The key is that all requests should return (even with errors)")
    print("  rather than crashing the backend.\n")
    
    input("Press Enter to start sequential test...")
    
    # Test 1: Sequential requests (baseline)
    seq_results = test_sequential_requests(session_id, num_requests=3)
    
    input("\nPress Enter to start concurrent test...")
    
    # Test 2: Concurrent requests
    concurrent_results = test_concurrent_requests(session_id, num_requests=5)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. If all concurrent requests completed, the threading lock works")
    print("2. Check 'nvidia-smi' during tests - memory should stay under 5.6GB")
    print("3. Backend logs will show requests being processed sequentially")
    print("4. Total time for concurrent ≈ sequential * num_requests (serialized)")

if __name__ == "__main__":
    main()
