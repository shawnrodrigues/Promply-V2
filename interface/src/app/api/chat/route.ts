import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the JSON data from the request
    const body = await request.json();
    
    // Forward the request to your Flask backend
    const response = await fetch('http://localhost:6969/chat', {
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
    console.error('Chat proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to AI service. Make sure Flask server is running on port 6969.' }, 
      { status: 500 }
    );
  }
}
