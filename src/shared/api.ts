import { Rule } from './Rule';
import { Reference } from './Reference';
import { McpConfig, CallToolResultWithElapsedTime } from '../main/mcp/types';
import { ChatSessionResponse, ChatState, MessageUpdate, ChatMessage } from './ChatSession';
import { WorkspaceWindow } from './workspace';
import { LLMProviderInfo, ILLMModel, LLMType } from './llm';
import { OpenDialogOptions, MessageBoxOptions } from 'electron';

export interface API {
  // Chat session management
  createChatTab: (tabId: string, modelProvider?: LLMType, modelId?: string) => Promise<ChatSessionResponse>;
  closeChatTab: (tabId: string) => Promise<ChatSessionResponse>;
  getChatState: (tabId: string) => Promise<ChatState | null>;
  sendMessage: (tabId: string, message: string | ChatMessage) => Promise<MessageUpdate>;
  clearModel: (tabId: string) => Promise<ChatSessionResponse>;
  switchModel: (tabId: string, modelType: string, modelId?: string) => Promise<ChatSessionResponse>;
  updateChatSettings: (tabId: string, settings: {
    maxChatTurns: number;
    maxOutputTokens: number;
    temperature: number;
    topP: number;
  }) => Promise<boolean>;

  // Chat context management
  addChatReference: (tabId: string, referenceName: string) => Promise<boolean>;
  removeChatReference: (tabId: string, referenceName: string) => Promise<boolean>;
  addChatRule: (tabId: string, ruleName: string) => Promise<boolean>;
  removeChatRule: (tabId: string, ruleName: string) => Promise<boolean>;

  // LLM Provider methods for model picker
  getProviderInfo: () => Promise<Record<string, LLMProviderInfo>>;
  validateProviderConfig: (provider: string) => Promise<{ isValid: boolean, error?: string }>;
  getModelsForProvider: (provider: string) => Promise<ILLMModel[]>;
  getInstalledProviders: () => Promise<string[]>;
  addProvider: (provider: string) => Promise<boolean>;
  removeProvider: (provider: string) => Promise<boolean>;
  getProviderConfig: (provider: string, key: string) => Promise<string | null>;
  setProviderConfig: (provider: string, key: string, value: string) => Promise<boolean>;

  // Settings API
  getSettingsValue: (key: string) => Promise<string | null>;
  setSettingsValue: (key: string, value: string) => Promise<boolean>;

  // Other existing methods
  getServerConfigs: () => Promise<McpConfig[]>;
  getMCPClient: (serverName: string) => Promise<{
    serverVersion: { name: string; version: string } | null;
    serverTools: any[];
    errorLog: string[];
  }>;
  callTool: (serverName: string, toolName: string, args: Record<string, unknown>) => Promise<CallToolResultWithElapsedTime>;
  toggleDevTools: () => Promise<boolean>;
  getSystemPrompt: () => Promise<string>;
  saveSystemPrompt: (prompt: string) => Promise<void>;
  showChatMenu: (hasSelection: boolean, x: number, y: number) => Promise<void>;
  showEditControlMenu: (editFlags: { 
    canUndo: boolean;
    canRedo: boolean; 
    canCut: boolean; 
    canCopy: boolean; 
    canPaste: boolean; 
    canSelectAll: boolean;
    x: number;
    y: number;
  }) => Promise<void>;
  openExternal: (url: string) => Promise<boolean>;
  getRules: () => Promise<Rule[]>;
  saveRule: (rule: Rule) => Promise<void>;
  deleteRule: (name: string) => Promise<void>;
  saveServerConfig: (server: McpConfig) => Promise<void>;
  reloadServerInfo: (serverName: string) => Promise<void>;
  deleteServerConfig: (serverName: string) => Promise<void>;
  getReferences: () => Promise<Reference[]>;
  saveReference: (reference: Reference) => Promise<void>;
  deleteReference: (name: string) => Promise<boolean>;
  pingServer: (name: string) => Promise<{ elapsedTimeMs: number }>;
  
  // Event listeners
  onRulesChanged: (callback: () => void) => () => void;
  offRulesChanged: (listener: () => void) => void;
  onReferencesChanged: (callback: () => void) => () => void;
  offReferencesChanged: (listener: () => void) => void;
  onProvidersChanged: (callback: () => void) => () => void;
  offProvidersChanged: (listener: () => void) => void;

  // Workspace methods
  showOpenDialog: (options: OpenDialogOptions) => Promise<{ canceled: boolean; filePaths: string[] }>;
  showMessageBox: (options: MessageBoxOptions) => Promise<{ response: number }>;
  getActiveWindows: () => Promise<WorkspaceWindow[]>;
  getRecentWorkspaces: () => Promise<string[]>;
  openWorkspace: (path: string) => Promise<void>;
  openInNewWindow: (path: string) => Promise<void>;
  createWorkspace: (windowId: string, path: string) => Promise<void>;
  createWorkspaceInNewWindow: (path: string) => Promise<void>;
  switchWorkspace: (windowId: string, workspacePath: string) => Promise<boolean>;
  focusWindow: (windowId: string) => Promise<boolean>;
  getCurrentWindowId: () => Promise<string>;
  cloneWorkspace: (sourcePath: string, targetPath: string) => Promise<{ success: boolean; error?: string; windowId?: string }>;
  workspaceExists: (path: string) => Promise<boolean>;
  onWorkspaceSwitched: (callback: (data: { windowId: string, workspacePath: string, targetWindowId: string }) => void) => (event: any, data: any) => void;
  offWorkspaceSwitched: (listener: (event: any, data: any) => void) => void;

  // App details
  getAppDetails: () => Promise<{ isPackaged: boolean }>;
}