import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    
    // Forward the request to your Flask backend
    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    });
    
    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`Flask server responded with status: ${response.status}`);
    }
    
    // Try to parse as JSON
    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Upload proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to upload file. Make sure Flask server is running on port 5000.' }, 
      { status: 500 }
    );
  }
}
