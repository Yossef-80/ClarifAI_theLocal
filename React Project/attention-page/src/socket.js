// socket.js
import { io } from 'socket.io-client';

export const socket = io('http://localhost:8000', {
  transports: ['websocket'],
});

socket.on('connect', () => {
  console.log('[Socket.IO] Connected:', socket.id);
});

socket.on('disconnect', () => {
  console.warn('[Socket.IO] Disconnected');
});

