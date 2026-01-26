import { act, renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { useDeleteFile, FileApiAction } from '../use-file-request';
import fileManagerService from '@/services/file-manager-service';

jest.mock('eventsource-parser/stream', () => ({}));

jest.mock('@/locales/config', () => ({}));

// Mock dependencies
jest.mock('@/services/file-manager-service', () => ({
  __esModule: true,
  default: {
    removeFile: jest.fn(),
  },
}));

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

jest.mock('@/components/ui/message', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
  },
}));

// Mock logic hooks
const mockSetPaginationParams = jest.fn();
const mockUseSetPaginationParams = jest.fn();

jest.mock('../route-hook', () => ({
  useSetPaginationParams: () => mockUseSetPaginationParams(),
}));

jest.mock('../logic-hooks', () => ({
  useGetPaginationWithRouter: jest.fn(() => ({
    pagination: { current: 1, pageSize: 10 },
    setPagination: jest.fn(),
  })),
  useHandleSearchChange: jest.fn(() => ({
    searchString: '',
    handleInputChange: jest.fn(),
  })),
}));

jest.mock('@/hooks/logic-hooks', () => ({
  useGetPaginationWithRouter: jest.fn(() => ({
    pagination: { current: 1, pageSize: 10 },
    setPagination: jest.fn(),
  })),
  useHandleSearchChange: jest.fn(() => ({
    searchString: '',
    handleInputChange: jest.fn(),
  })),
}));

jest.mock('react-router', () => ({
  useSearchParams: () => [new URLSearchParams({ folderId: 'root' })],
}));

describe('useDeleteFile', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          gcTime: Infinity,
        },
      },
    });
    mockSetPaginationParams.mockClear();
    (fileManagerService.removeFile as jest.Mock).mockResolvedValue({
      data: { code: 0 },
    });

    mockUseSetPaginationParams.mockReturnValue({
        setPaginationParams: mockSetPaginationParams,
        page: 1,
        size: 10
    });
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it('should use ACTIVE query to calculate pagination (fix for search context bug)', async () => {
    mockUseSetPaginationParams.mockReturnValue({
        setPaginationParams: mockSetPaginationParams,
        page: 2,
        size: 10
    });

    // Inactive query (e.g. stale full list), total 100.
    queryClient.setQueryData(
      [FileApiAction.FetchFileList, { id: 'root', debouncedSearchString: '', current: 2, pageSize: 10 }],
      { files: [], total: 100, parent_folder: {} }
    );

    const InnerActiveQuery = () => {
        useQuery({
             queryKey: [FileApiAction.FetchFileList, { id: 'root', debouncedSearchString: 'foo', current: 2, pageSize: 10 }],
             queryFn: () => ({ files: [], total: 1, parent_folder: {} }),
             initialData: { files: [], total: 1, parent_folder: {} }
        });
        return null;
    };

    const WrapperWithActiveQuery = ({ children }: { children: React.ReactNode }) => {
        return (
            <QueryClientProvider client={queryClient}>
                <InnerActiveQuery />
                {children}
            </QueryClientProvider>
        );
    };

    const { result } = renderHook(() => useDeleteFile(), { wrapper: WrapperWithActiveQuery });

    await act(async () => {
      await result.current.deleteFile({ fileIds: ['1'], parentId: 'root' });
    });

    // If it used total 100: newTotal 99, maxPage 10. Page 2 is valid. Call count 0.
    // If it used total 1: newTotal 0, maxPage 1 (Math.max(1, 0)). Page 2 > 1. Call count 1 (arg 1).
    expect(mockSetPaginationParams).toHaveBeenCalledWith(1);
  });

  it('should NOT reset to page 1 if page is still valid (active query check)', async () => {
     mockUseSetPaginationParams.mockReturnValue({
        setPaginationParams: mockSetPaginationParams,
        page: 2,
        size: 10
    });

    const InnerActiveQuery = () => {
        useQuery({
             queryKey: [FileApiAction.FetchFileList, { id: 'root', debouncedSearchString: 'bar', current: 2, pageSize: 10 }],
             queryFn: () => ({ files: [], total: 12, parent_folder: {} }),
             initialData: { files: [], total: 12, parent_folder: {} }
        });
        return null;
    };

    const WrapperWithActiveQuery = ({ children }: { children: React.ReactNode }) => {
        return (
            <QueryClientProvider client={queryClient}>
                <InnerActiveQuery />
                {children}
            </QueryClientProvider>
        );
    };

    const { result } = renderHook(() => useDeleteFile(), { wrapper: WrapperWithActiveQuery });

    await act(async () => {
      await result.current.deleteFile({ fileIds: ['1'], parentId: 'root' });
    });

    expect(mockSetPaginationParams).not.toHaveBeenCalled();
  });
});
