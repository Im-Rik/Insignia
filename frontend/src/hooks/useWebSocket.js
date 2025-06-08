// src/hooks/useWebSocket.js

import { useState, useEffect, useRef } from 'react';

/**
 * A custom hook to manage a WebSocket connection.
 *
 * @param {string} url - The WebSocket server URL.
 * @returns {{ sendMessage: function, isConnected: boolean }}
 */
const useWebSocket = (url) => {
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    // Do not connect if the URL is not provided
    if (!url) return;

    // Initialize WebSocket connection
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log('WebSocket connection established.');
      setIsConnected(true);
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed.');
      setIsConnected(false);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    // Clean up the connection when the component unmounts or the URL changes
    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, [url]);

  /**
   * Sends a message through the WebSocket.
   * @param {string} eventName - The name of the event/topic to emit.
   * @param {any} data - The payload to send.
   */
  const sendMessage = (eventName, data) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ event: eventName, data });
      socketRef.current.send(message);
    } else {
      console.warn('Cannot send message, WebSocket is not connected.');
    }
  };

  return { sendMessage, isConnected };
};

export default useWebSocket;
