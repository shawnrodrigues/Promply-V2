import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    const file = formData.get('pdf') || formData.get('file');
    const fileName = file && typeof file === 'object' && 'name' in file ? String((file as File).name) : 'unknown';
    console.log(`[api/upload] Forwarding upload to Flask: ${fileName}`);
    
    // Forward the request to your Flask backend
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300000); // 5 min timeout for large OCR jobs
    const response = await fetch('http://localhost:6969/upload', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');

    let payload: any = null;
    if (isJson) {
      payload = await response.json();
    } else {
      const text = await response.text();
      payload = {
        error: text?.slice(0, 1000) || `Flask server responded with status ${response.status}`
      };
    }

    console.log(`[api/upload] Flask status ${response.status}; ok=${response.ok}`);

    // Pass through Flask errors instead of masking them as a proxy crash.
    if (!response.ok) {
      return NextResponse.json(
        {
          status: 'error',
          error:
            payload?.error ||
            payload?.message ||
            `Flask server responded with status ${response.status}`,
          backendStatus: response.status,
        },
        { status: response.status }
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    console.error('Upload proxy error:', error);
    return NextResponse.json(
      {
        status: 'error',
        error: error instanceof Error
          ? error.message
          : 'Failed to upload file. Make sure Flask server is running on port 6969.'
      },
      { status: 500 }
    );
  }
}
