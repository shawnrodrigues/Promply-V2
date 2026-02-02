const { spawn } = require('child_process');
const { exec } = require('child_process');

// Start Next.js dev server
const nextProcess = spawn('npm', ['run', 'dev:next'], {
  stdio: 'inherit',
  shell: true
});

// Wait a moment for server to start, then open browser
setTimeout(() => {
  const url = 'http://localhost:3000';
  console.log(`\nOpening browser to ${url}...\n`);
  
  // Cross-platform browser opening
  const command = process.platform === 'win32' 
    ? `start ${url}` 
    : process.platform === 'darwin' 
    ? `open ${url}` 
    : `xdg-open ${url}`;
  
  exec(command);
}, 2000);

// Handle exit
process.on('SIGINT', () => {
  nextProcess.kill('SIGINT');
  process.exit();
});

process.on('SIGTERM', () => {
  nextProcess.kill('SIGTERM');
  process.exit();
});
