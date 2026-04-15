import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the JSON data from the request
    const body = await request.json();
    const query = typeof body?.query === 'string' ? body.query : '';
    console.log(`[api/chat] Forwarding query to Flask: ${query.slice(0, 120)}`);
    
    // Forward the request to your Flask backend
    const response = await fetch('http://localhost:6969/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Try to parse as JSON
    const data = await response.json();
    console.log(`[api/chat] Flask status ${response.status}; response length ${JSON.stringify(data).length}`);

    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`Flask server responded with status: ${response.status}`);
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Chat proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to AI service. Make sure Flask server is running on port 6969.' }, 
      { status: 500 }
    );
  }
}
