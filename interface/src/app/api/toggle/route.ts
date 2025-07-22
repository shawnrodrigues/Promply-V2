import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the JSON data from the request
    const body = await request.json();
    
    // Forward the request to your Flask backend
    const response = await fetch('http://localhost:5000/toggle', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`Flask server responded with status: ${response.status}`);
    }
    
    // Try to parse as JSON
    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Toggle proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to toggle mode. Make sure Flask server is running on port 5000.' }, 
      { status: 500 }
    );
  }
}
