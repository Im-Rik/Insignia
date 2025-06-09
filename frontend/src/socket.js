import { io } from 'socket.io-client';

// IMPORTANT: Make sure this URL matches your Python backend server address.
const SOCKET_URL = 'http://localhost:5000';

export const socket = io(SOCKET_URL, {
  autoConnect: false, // We will connect manually from our component.
  transports: ['websocket'], // Prefer WebSocket transport.
});